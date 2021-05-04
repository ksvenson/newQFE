// metropolis.cc

// ref: N. Metropolis, et al., J. Chem. Phys. 21, 1087 (1953).

#include "metropolis.h"

#include "lattice.h"

void QfeMetropolis::Init(QfeLattice* lattice) {
  this->lattice = lattice;
  z = 0.5;
}

double QfeMetropolis::Update() {
  double accept = 0.0;
  for (int s = 0; s < lattice->n_sites(); s++) {
    accept += UpdateSite(s);
  }
  return accept / lattice->n_sites();
}

double QfeMetropolis::UpdateSite(int s) {
  double phi_old = lattice->phi[s];
  double phi_old2 = phi_old * phi_old;
  double phi_old4 = phi_old2 * phi_old2;

  // generate a new field value
  double phi_new = phi_old + lattice->rng.RandReal(-1.0, 1.0) * z;
  double phi_new2 = phi_new * phi_new;
  double phi_new4 = phi_new2 * phi_new2;

  double delta_phi = phi_new - phi_old;
  double delta_phi2 = phi_new2 - phi_old2;
  double delta_phi4 = phi_new4 - phi_old4;

  double delta_action = 0.0;

  // kinetic contribution to the action
  QfeSite site = lattice->sites[s];
  for (int n = 0; n < site.nn; n++) {
    double link_wt = lattice->links[site.links[n]].wt;
    delta_action -= lattice->phi[site.neighbors[n]] * delta_phi * link_wt;
    delta_action += 0.5 * delta_phi2 * link_wt;
  }

  // musq and lambda contributions to the action
  double mass_term = -0.5 * lattice->musq * delta_phi2;
  double interaction_term = lattice->lambda * delta_phi4;
  delta_action += (mass_term + interaction_term) * site.wt;

  // metropolis algorithm
  if (delta_action <= 0.0 || lattice->rng.RandReal() < exp(-delta_action)) {
    lattice->phi[s] = phi_new;
    return 1.0;
  } else {
    return 0.0;
  }
}
