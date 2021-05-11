// overrelax.cc

// overrelaxation update algorithm
// ref: S. L. Adler, Phys. Rev. D 23, 2901 (1981).
// C. Whitmer, Phys. Rev. D 29, 306 (1984).
// F. R. Brown and T. J. Woch, Phys. Rev. Lett. 58, 2394 (1987).
// M. Creutz, Phys. Rev. D 36, 515 (1987).
// M. Hasenbush, J. Phys. A: Math. Gen. 32 4851 (1999).

#include "overrelax.h"

#include "lattice.h"

QfeOverrelax::QfeOverrelax(QfeLattice* lattice) {
  this->lattice = lattice;
  demon = 0.0;
}

// first we find the value of phi that minimizes the quadratic part of the
// action this site. in other words, set lambda = 0 and solve dS/dphi_x = 0.
// call this value phi_min. to minimize the action we would just change phi to
// phi_min, but the overrelaxation trick is to change phi by a little bit more.
// in the literature, this is normally parametrized as
//     phi -> phi + omega * delta_phi
// where delta_phi is phi_min - phi. we set omega to 2, which actually keeps
// the quadratic part of the action invariant. then we use a "demon" (as
// described in the Hasenbush reference) to deal with the non-quadratic part
// of the action. The accept-reject step ensures that we maintain detailed
// balance.

double QfeOverrelax::Update() {
  int accept = 0;
  for (int s = 0; s < lattice->n_sites(); s++) {
    accept += UpdateSite(s);
  }
  return double(accept) / double(lattice->n_sites());
}

int QfeOverrelax::UpdateSite(int s) {
  QfeSite* site = &lattice->sites[s];
  double old_phi = lattice->phi[s];

  double numerator = 0.0;
  double denominator = -lattice->musq * site->wt;
  for (int n = 0; n < site->nn; n++) {
    double link_wt = lattice->links[site->links[n]].wt;
    numerator += link_wt * lattice->phi[site->neighbors[n]];
    denominator += link_wt;
  }
  double new_phi = 2.0 * numerator / denominator - old_phi;

  double old_phi4 = old_phi * old_phi * old_phi * old_phi;
  double new_phi4 = new_phi * new_phi * new_phi * new_phi;
  double new_demon = demon;
  new_demon += site->wt * lattice->lambda * (old_phi4 - new_phi4);

  if (new_demon >= 0) {
    lattice->phi[s] = new_phi;
    demon = new_demon;
    return 1;
  } else {
    return 0;
  }
}
