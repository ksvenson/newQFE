// lattice.cc

#include "lattice.h"

#include <vector>

QfeLattice::QfeLattice() {
  musq = 1.2725;
  lambda = 0.25;
}

// skew is a number between 0 and 1 which determines how skewed the triangles
// are. a value of 0 corresponds to equilateral triangles, and a value of 1
// corresponds to right triangles, which is equivalent to a square lattice
// because the diagonal link weights will be zero.

void QfeLattice::InitTriangle(int N, double skew) {

  // create sites
  sites.resize(N * N);
  phi.resize(n_sites());

  for (int s = 0; s < n_sites(); s++) {
    sites[s].id = s;
    sites[s].wt = 1.0;
  }

  // create links
  links.clear();

  // if skew = 0.0, all weights are the same
  // if skew = 1.0, the middle link weight is zero
  // average weight is 2/3
  double wt1 = (2.0 + skew) / 3.0;
  double wt2 = (2.0 - 2.0 * skew) / 3.0;
  double wt3 = wt1;

  for (int s = 0; s < n_sites(); s++) {
    int x = s % N;
    int y = s / N;

    // add links in the "forward" direction (3 links)
    int x_right = (x + 1) % N;
    int y_down = (y + 1) % N;
    AddLink(s, x_right + y * N, wt1);
    AddLink(s, x + y_down * N, wt2);
    AddLink(s, x_right + y_down * N, wt3);
  }
}

// add a link from site a to b with weight wt

QfeLink QfeLattice::AddLink(int a, int b, double wt) {
  QfeLink link;
  link.id = n_links();
  link.wt = wt;
  link.sites[0] = a;
  link.sites[1] = b;

  links.push_back(link);
  sites[a].links.push_back(link.id);
  sites[a].neighbors.push_back(b);
  sites[b].links.push_back(link.id);
  sites[b].neighbors.push_back(a);
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
    mag += phi1;
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
