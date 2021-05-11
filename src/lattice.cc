// lattice.cc

#include "lattice.h"

#include <vector>

QfeLattice::QfeLattice(double musq, double lambda) {
  this->musq = musq;
  this->lambda = lambda;
}

// skew is a number between 0 and 1 which determines how skewed the triangles
// are. a value of 0 corresponds to equilateral triangles, and a value of 1
// corresponds to right triangles, which is equivalent to a square lattice
// because the diagonal link weights will be zero.

void QfeLattice::InitTriangle(int N, double skew) {

  // create sites
  sites.resize(N * N);
  phi.resize(n_sites());

  // set all site weights to 1.0
  for (int s = 0; s < n_sites(); s++) {
    sites[s].wt = 1.0;
    sites[s].nn = 0;
  }

  // create links
  links.clear();

  // if skew = 0.0, all weights are the same (equilateral triangles)
  // if skew = 1.0, the middle link weight is zero (right triangles)
  // average weight is 2/3
  double wt1 = (2.0 + skew) / 3.0;
  double wt2 = (2.0 - 2.0 * skew) / 3.0;
  double wt3 = wt1;

  for (int s = 0; s < n_sites(); s++) {
    int x = s % N;
    int y = s / N;

    // add links in the "forward" direction (3 links per site)
    // each link will end up with 6 neighbors
    int xp1 = (x + 1) % N;
    int yp1 = (y + 1) % N;
    AddLink(s, xp1 + y * N, wt1);
    AddLink(s, x + yp1 * N, wt2);
    AddLink(s, xp1 + yp1 * N, wt3);
  }
}

// add a link from site a to b with weight wt

QfeLink QfeLattice::AddLink(int a, int b, double wt) {

  int l = n_links();  // link index
  QfeLink link;
  link.wt = wt;

  int nn_a = sites[a].nn;
  sites[a].neighbors[nn_a] = b;
  sites[a].links[nn_a] = l;
  sites[a].nn++;
  link.sites[0] = a;

  int nn_b = sites[b].nn;
  sites[b].neighbors[nn_b] = a;
  sites[b].links[nn_b] = l;
  sites[b].nn++;
  link.sites[1] = b;

  links.push_back(link);
  return link;
}

double QfeLattice::Action() {
  double action = 0.0;

  // kinetic contribution
  for (int l = 0; l < n_links(); l++) {
    int a = links[l].sites[0];
    int b = links[l].sites[1];
    double delta_phi = phi[a] - phi[b];
    double delta_phi2 = delta_phi * delta_phi;
    action += 0.5 * delta_phi2 * links[l].wt;
  }

  // musq and lambda contributions
  mag = 0.0;
  for (int s = 0; s < n_sites(); s++) {
    double phi1 = phi[s];
    double phi2 = phi1 * phi1;  // phi^2
    double phi4 = phi2 * phi2;  // phi^4
    mag += phi1 * sites[s].wt;
    double mass_term = -0.5 * musq * phi2;
    double interaction_term = lambda * phi4;
    action += (mass_term + interaction_term) * sites[s].wt;
  }
  mag /= n_sites();

  return action / n_sites();
}

void QfeLattice::HotStart() {
  for (int s = 0; s < n_sites(); s++) {
    phi[s] = rng.RandNormal();
  }
}

void QfeLattice::ColdStart() {
  std::fill(phi.begin(), phi.end(), 0.0);
}
