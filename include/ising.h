// ising.h

#pragma once

#include <cmath>
#include <stack>
#include <vector>
#include "lattice.h"

class QfeIsing {

public:
  QfeIsing(QfeLattice* lattice, double beta);
  double Action();
  double MeanSpin();
  void HotStart();
  void ColdStart();
  double Metropolis();
  int WolffUpdate();

  QfeLattice* lattice;
  std::vector<double> spin;  // Z2 field
  std::vector<double> beta_ct;  // local beta counterterm
  double beta;  // bare coupling

  std::vector<bool> is_clustered;  // keeps track of which sites are clustered
  std::vector<int> wolff_cluster;  // array of clustered sites
};

QfeIsing::QfeIsing(QfeLattice* lattice, double beta) {
  this->lattice = lattice;
  this->beta = beta;
  spin.resize(lattice->sites.size());
  beta_ct.resize(lattice->links.size(), 0.0);
  is_clustered.resize(lattice->sites.size());
}

double QfeIsing::Action() {
  double action = 0.0;

  // sum over links
  for (int l = 0; l < lattice->n_links; l++) {
    QfeLink* link = &lattice->links[l];
    int a = link->sites[0];
    int b = link->sites[1];
    action -= (beta + beta_ct[l]) * spin[a] * spin[b] * link->wt;
  }

  return action / double(lattice->n_sites);
}

double QfeIsing::MeanSpin() {
  double m = 0.0;
  for (int s = 0; s < lattice->n_sites; s++) {
    m += spin[s] * lattice->sites[s].wt;
  }
  return m / double(lattice->n_sites);
}

void QfeIsing::HotStart() {
  for (int s = 0; s < lattice->n_sites; s++) {
    spin[s] = double(lattice->rng.RandInt(0, 1) * 2 - 1);
  }
}

void QfeIsing::ColdStart() {
  std::fill(spin.begin(), spin.begin() + lattice->n_sites, 1.0);
}

// metropolis update algorithm
// ref: N. Metropolis, et al., J. Chem. Phys. 21, 1087 (1953).

double QfeIsing::Metropolis() {
  int accept = 0;
  for (int s = 0; s < lattice->n_sites; s++) {
    double delta_S = 0.0;

    // sum over links connected to this site
    QfeSite* site = &lattice->sites[s];
    for (int n = 0; n < site->nn; n++) {
      int l = site->links[n];
      double link_wt = lattice->links[l].wt;
      delta_S += (beta + beta_ct[l]) * spin[site->neighbors[n]] * link_wt;
    }
    delta_S *= 2.0 * spin[s];

    // metropolis algorithm
    if (delta_S <= 0.0 || lattice->rng.RandReal() < exp(-delta_S)) {
      spin[s] *= -1.0;
      accept++;
    }
  }
  return double(accept) / double(lattice->n_sites);
}

// wolff cluster update algorithm
// ref: U. Wolff, Phys. Rev. Lett. 62, 361 (1989).

int QfeIsing::WolffUpdate() {

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

    // flip the spin
    double value = spin[s];
    spin[s] = -value;

    // try to add neighbors
    QfeSite* site = &lattice->sites[s];
    for (int n = 0; n < site->nn; n++) {
      int l = site->links[n];
      double link_wt = lattice->links[l].wt;
      s = site->neighbors[n];

      // skip if the site is already clustered
      if (is_clustered[s]) continue;

      // skip if sign bits don't match
      if (std::signbit(value) != std::signbit(spin[s])) continue;

      double prob = 1.0 - exp(-2.0 * (beta + beta_ct[l]) * link_wt);
      if (lattice->rng.RandReal() < prob) {
        // add the site to the cluster
        wolff_cluster.push_back(s);
        is_clustered[s] = true;
        stack.push(s);
      }
    }
  }

  return wolff_cluster.size();
}
