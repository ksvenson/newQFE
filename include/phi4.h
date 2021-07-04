// phi4.h

#pragma once

#include <cassert>
#include <cmath>
#include <stack>
#include <vector>
#include <Eigen/Sparse>
#include "lattice.h"

class QfePhi4 {

public:
  QfePhi4(QfeLattice* lattice, double msq, double lambda);
  double Action();
  double MeanPhi();
  void HotStart();
  void ColdStart();
  double Metropolis();
  double Overrelax();
  int WolffUpdate();
  std::vector<double> MInverse(double m0);

  QfeLattice* lattice;
  std::vector<double> phi;  // scalar field
  std::vector<double> msq_ct;  // local mass counter terms
  std::vector<double> moments;  // magnetic moments
  double lambda;  // bare coupling
  double msq;  // bare mass squared

  double metropolis_z;
  double overrelax_demon;
  std::vector<bool> is_clustered;  // keeps track of which sites are clustered
  std::vector<int> wolff_cluster;  // array of clustered sites
};

QfePhi4::QfePhi4(QfeLattice* lattice, double msq, double lambda) {
  this->lattice = lattice;
  this->msq = msq;
  this->lambda = lambda;
  metropolis_z = 0.5;
  overrelax_demon = 0.0;
  phi.resize(lattice->sites.size(), 0.0);
  msq_ct.resize(lattice->sites.size(), 0.0);
  is_clustered.resize(lattice->sites.size());
}

double QfePhi4::Action() {
  double action = 0.0;

  // kinetic contribution
  for (int l = 0; l < lattice->n_links; l++) {
    int a = lattice->links[l].sites[0];
    int b = lattice->links[l].sites[1];
    double delta_phi = phi[a] - phi[b];
    double delta_phi2 = delta_phi * delta_phi;
    action += 0.5 * delta_phi2 * lattice->links[l].wt;
  }

  // msq and lambda contributions
  for (int s = 0; s < lattice->n_sites; s++) {
    double phi1 = phi[s];
    double phi2 = phi1 * phi1;  // phi^2
    double phi4 = phi2 * phi2;  // phi^4
    double mass_term = 0.5 * (msq + msq_ct[s]) * phi2;
    double interaction_term = lambda * phi4;
    action += (mass_term + interaction_term) * lattice->sites[s].wt;
  }

  return action / double(lattice->n_sites);
}

double QfePhi4::MeanPhi() {
  double m = 0.0;
  for (int s = 0; s < lattice->n_sites; s++) {
    m += phi[s] * lattice->sites[s].wt;
  }
  return m / double(lattice->n_sites);
}

void QfePhi4::HotStart() {
  for (int s = 0; s < lattice->n_sites; s++) {
    phi[s] = lattice->rng.RandNormal();
  }
}

void QfePhi4::ColdStart() {
  std::fill(phi.begin(), phi.begin() + lattice->n_sites, 0.0);
}

// metropolis update algorithm
// ref: N. Metropolis, et al., J. Chem. Phys. 21, 1087 (1953).

double QfePhi4::Metropolis() {
  int accept = 0;
  for (int s = 0; s < lattice->n_sites; s++) {
    double phi_old = phi[s];
    double phi_old2 = phi_old * phi_old;
    double phi_old4 = phi_old2 * phi_old2;

    // generate a new field value
    double phi_new = phi_old + lattice->rng.RandReal(-1.0, 1.0) * metropolis_z;
    double phi_new2 = phi_new * phi_new;
    double phi_new4 = phi_new2 * phi_new2;

    double delta_phi = phi_new - phi_old;
    double delta_phi2 = phi_new2 - phi_old2;
    double delta_phi4 = phi_new4 - phi_old4;

    double delta_S = 0.0;

    // kinetic contribution to the action
    QfeSite* site = &lattice->sites[s];
    for (int n = 0; n < site->nn; n++) {
      int l = site->links[n];
      double link_wt = lattice->links[l].wt;
      delta_S -= phi[site->neighbors[n]] * delta_phi * link_wt;
      delta_S += 0.5 * delta_phi2 * link_wt;
    }

    // msq and lambda contributions to the action
    double mass_term = 0.5 * (msq + msq_ct[s]) * delta_phi2;
    double interaction_term = lambda * delta_phi4;
    delta_S += (mass_term + interaction_term) * site->wt;

    // metropolis algorithm
    if (delta_S <= 0.0 || lattice->rng.RandReal() < exp(-delta_S)) {
      phi[s] = phi_new;
      accept++;
    }
  }
  return double(accept) / double(lattice->n_sites);
}

// overrelaxation update algorithm
// ref: S. L. Adler, Phys. Rev. D 23, 2901 (1981).
// C. Whitmer, Phys. Rev. D 29, 306 (1984).
// F. R. Brown and T. J. Woch, Phys. Rev. Lett. 58, 2394 (1987).
// M. Creutz, Phys. Rev. D 36, 515 (1987).
// M. Hasenbusch, J. Phys. A: Math. Gen. 32 4851 (1999).
//
// first we find the value of phi that minimizes the quadratic part of the
// action at this site. in other words, set lambda = 0 and solve dS/dphi_x = 0.
// call this value phi_min. to minimize the action we would just change phi to
// phi_min, but the overrelaxation trick is to change phi by a little bit more.
// in the literature, this is normally parametrized as
//     phi -> phi + omega * delta_phi
// where delta_phi is phi_min - phi. we set omega to 2, which actually keeps
// the quadratic part of the action invariant. then we use a "demon" (as
// described in the Hasenbusch reference) to deal with the non-quadratic part
// of the action. The accept-reject step ensures that we maintain detailed
// balance.

double QfePhi4::Overrelax() {
  int accept = 0;
  for (int s = 0; s < lattice->n_sites; s++) {
    QfeSite* site = &lattice->sites[s];
    double phi_old = phi[s];

    double numerator = 0.0;
    double denominator = (msq + msq_ct[s]) * site->wt;
    for (int n = 0; n < site->nn; n++) {
      double link_wt = lattice->links[site->links[n]].wt;
      numerator += link_wt * phi[site->neighbors[n]];
      denominator += link_wt;
    }
    double phi_new = 2.0 * numerator / denominator - phi_old;

    double phi_old4 = phi_old * phi_old * phi_old * phi_old;
    double phi_new4 = phi_new * phi_new * phi_new * phi_new;
    double new_demon = overrelax_demon;
    new_demon += site->wt * lambda * (phi_old4 - phi_new4);

    if (new_demon >= 0) {
      phi[s] = phi_new;
      overrelax_demon = new_demon;
      accept++;
    }
  }
  return double(accept) / double(lattice->n_sites);
}

// wolff cluster update algorithm
// ref: U. Wolff, Phys. Rev. Lett. 62, 361 (1989).

int QfePhi4::WolffUpdate() {

  // remove all sites from the cluster
  std::fill(is_clustered.begin(), is_clustered.end(), false);
  wolff_cluster.clear();

  // create the stack
  std::stack<int> stack;

  // choose a random site and add it to the cluster
  int s = lattice->rng.RandInt(0, lattice->n_sites - 1);
  wolff_cluster.push_back(s);
  is_clustered[s] = true;
  stack.push(s);

  while (stack.size() != 0) {
    s = stack.top();
    stack.pop();

    // try to add neighbors
    QfeSite* site = &lattice->sites[s];
    double value = phi[s];
    for (int n = 0; n < site->nn; n++) {
      int l = site->links[n];
      double link_wt = lattice->links[l].wt;
      s = site->neighbors[n];

      // skip if the site is already clustered
      if (is_clustered[s]) continue;

      // skip if phi is zero at this site. this will most likely be a dummy
      // site at a dirichlet boundary, but it's still correct even if it's
      // not because the probability of adding it to the cluster will be zero
      if (phi[s] == 0.0) continue;

      // skip if sign bits don't match
      if (std::signbit(value) != std::signbit(phi[s])) continue;

      double prob = 1.0 - exp(-2.0 * value * phi[s] * link_wt);
      if (lattice->rng.RandReal() < prob) {
        // add the site to the cluster
        wolff_cluster.push_back(s);
        is_clustered[s] = true;
        stack.push(s);
      }
    }
  }

  for (int s = 0; s < wolff_cluster.size(); s++) {
    phi[wolff_cluster[s]] *= -1.0;
  }

  return wolff_cluster.size();
}

std::vector<double> QfePhi4::MInverse(double m0) {
  std::vector<Eigen::Triplet<double>> M_elements;

  // add nearest-neighbor interaction terms
  for (int l = 0; l < lattice->n_links; l++) {
    QfeLink* link = &lattice->links[l];
    int a = link->sites[0];
    int b = link->sites[1];
    M_elements.push_back(Eigen::Triplet<double>(a, b, -link->wt));
    M_elements.push_back(Eigen::Triplet<double>(b, a, -link->wt));
  }

  // add self-interaction terms
  for (int s = 0; s < lattice->n_sites; s++) {
    QfeSite* site = &lattice->sites[s];
    double wt_sum = m0 * site->wt / double(lattice->n_sites);
    for (int n = 0; n < site->nn; n++) {
      int l = site->links[n];
      wt_sum += lattice->links[l].wt;
    }
    M_elements.push_back(Eigen::Triplet<double>(s, s, wt_sum));
  }

  Eigen::SparseMatrix<double> M(lattice->n_sites, lattice->n_sites);
  M.setFromTriplets(M_elements.begin(), M_elements.end());

  // create the solver and analyze M
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cg;
  cg.compute(M);
  assert(cg.info() == Eigen::Success);

  std::map<int, double> ct_map;
  std::vector<double> M_inv(lattice->n_sites);
  for (int s = 0; s < lattice->n_sites; s++) {

    // assume sites with the same weight have the same M_inv
    int wt_int = int(round(lattice->sites[s].wt * 1.0e10));
    if (ct_map.find(wt_int) == ct_map.end()) {

      // create a source and solve via conjugate gradient
      Eigen::VectorXd b = Eigen::VectorXd::Zero(lattice->n_sites);
      b(s) = 1.0;
      Eigen::VectorXd x = cg.solve(b);
      assert(cg.info() == Eigen::Success);

      // save M_inv in the map
      ct_map[wt_int] = x(s);
    }

    // set the counterterm
    M_inv[s] = ct_map[wt_int];
  }

  return M_inv;
}
