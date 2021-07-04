// ads3.h

#pragma once

#include <vector>
#include "ads2.h"

class QfeLatticeAdS3 : public QfeLatticeAdS2 {

public:
  QfeLatticeAdS3(int n_layers, int q, int Nt = 0);
  double Sigma(int s1, int s2);
  double DeltaT(int s1, int s2);

  int Nt;
  double t_scale;  // ratio of temporal to spatial lattice spacing
  std::vector<int> t;  // time coordinate of sites
};

/**
 * @brief Initialize a triangulated lattice on AdS3.
 *
 * A Poincaré disk is used to generate an AdS2 slice of the lattice. Then
 * additional slices are created and spaced appropriately in the time
 * direction.
 *
 * [1] R. Brower et al., Phys. Rev. D, 103, 094507 (2021).
 * @see https://arxiv.org/abs/1912.07606
 *
 * @param n_layers Number of layers to create
 * @param q Number of triangles meeting at each site (should be greater than 6)
 * @param Nt Number of time slices
 */

QfeLatticeAdS3::QfeLatticeAdS3(int n_layers, int q, int Nt) :
  QfeLatticeAdS2(n_layers, q) {

  // set number of time slices, default equal to number of boundary sites
  if (Nt == 0) {
    Nt = boundary_sites.size();
  }
  this->Nt = Nt;

  // set temporal-spatial lattice spacing ratio (speed of light)
  // t_scale = 1.0;  // fixed spacing
  // t_scale = 1.0 / layer_rho[1];
  // t_scale = layer_cosh_rho[1];  // uniform spacing at center
  // t_scale = double(n_layers) / (layer_cosh_rho[n_layers] * M_PI);
  // t_scale = layer_cosh_rho[n_layers];  // uniform spacing at boundary
  // t_scale = total_cosh_rho[n_layers];  // uniform (average) spacing everywhere

  // equal geodesic spacing in time and theta directions at boundary
  double tanh2_sum = 0.0;
  double cos_theta_sum = 0.0;
  for (int i = 0; i < n_boundary; i++) {
    int s = boundary_sites[i];
    int sp1 = boundary_sites[(i + 1) % n_boundary];
    double tanh_rho = tanh(rho[s]);
    tanh2_sum += tanh_rho * tanh_rho;
    cos_theta_sum += cos(Theta(s, sp1));
  }
  double tanh2_mean = tanh2_sum / double(n_boundary);
  double cos_theta_mean = cos_theta_sum / double(n_boundary);
  t_scale = 1.0 / acosh(1.0 + tanh2_mean * (1.0 - cos_theta_mean));

  // we start with a single AdS2 slice, and we need to make Nt copies
  int n_sites_slice = layer_offset[n_layers + 1];  // dynamic sites per slice
  n_sites = n_sites_slice * Nt;  // number of dynamic sites
  sites.resize(n_sites + n_dummy);
  site_layers.resize(sites.size());

  // resize coordinate arrays
  z.resize(sites.size());
  r.resize(sites.size());
  theta.resize(sites.size());
  rho.resize(sites.size());
  u.resize(sites.size());
  t.resize(sites.size(), 0);

  // move the dummy sites to the end of the array
  layer_sites[n_layers + 1].clear();
  for (int i = 0; i < n_dummy; i++) {
    int s_old = n_sites_slice + i;
    int s_new = n_sites + i;
    sites[s_new].nn = 0;
    sites[s_new].wt = cosh(rho[s_old]);
    sites[s_new].id = sites[s_old].id;
    layer_sites[n_layers + 1].push_back(s_new);

    // copy coordinates
    z[s_new] = z[s_old];
    r[s_new] = r[s_old];
    theta[s_new] = theta[s_old];
    rho[s_new] = rho[s_old];
    u[s_new] = u[s_old];
    t[s_new] = 0;  // all dummy sites are at t = 0
  }

  // duplicate sites on the other slices
  for (int s0 = 0; s0 < n_sites_slice; s0++) {

    // the original AdS2 site on the t = 0 slice
    QfeSite* site0 = &sites[s0];

    // adjust the site weight for AdS3
    site0->wt = cosh(rho[s0]) / t_scale;
    int layer = site_layers[s0];  // the layer that this site is on

    // adjust neighbor table for dummy neighbors
    for (int n = 0; n < site0->nn; n++) {
      int s = site0->neighbors[n];
      if (s >= n_sites_slice) {
        site0->neighbors[n] = n_sites + (s - n_sites_slice);
      }
    }

    // copy site coordinates and weights for sites on the other time slices
    for (int tt = 1; tt < Nt; tt++) {

      int s = n_sites_slice * tt + s0;  // calculate site index
      sites[s].nn = 0;  // add links later
      sites[s].wt = site0->wt;  // site weight is the same
      sites[s].id = site0->id;

      // set the site's layer and add it to the bulk or boundary
      site_layers[s] = layer;
      layer_sites[layer].push_back(s);
      if (layer < n_layers) {
        bulk_sites.push_back(s);
      } else if (layer == n_layers) {
        boundary_sites.push_back(s);
      }

      // set the site's coordinates
      z[s] = z[s0];
      r[s] = r[s0];
      theta[s] = theta[s0];
      rho[s] = rho[s0];
      u[s] = u[s0];
      t[s] = tt;
    }
  }

  // update the number of bulk/boundary sites
  n_bulk = bulk_sites.size();
  n_boundary = boundary_sites.size();

  // duplicate links on the other slices
  int n_links_slice = links.size();
  for (int l = 0; l < n_links_slice; l++) {

    int s_a = links[l].sites[0];
    int s_b = links[l].sites[1];

    // adjust neighbor table for dummy neighbors
    if (s_a >= n_sites_slice) {
      s_a = n_sites + (s_a - n_sites_slice);
      links[l].sites[0] = s_a;
    }
    if (s_b >= n_sites_slice) {
      s_b = n_sites + (s_b - n_sites_slice);
      links[l].sites[1] = s_b;
    }

    // adjust the link weight
    double link_wt = 0.5 * (cosh(rho[s_a]) + cosh(rho[s_b])) / t_scale;
    links[l].wt = link_wt;

    // add links in the other slices
    for (int tt = 1; tt < Nt; tt++) {
      if (s_a < n_sites) {
        s_a = (s_a + n_sites_slice) % n_sites;
      }
      if (s_b < n_sites) {
        s_b = (s_b + n_sites_slice) % n_sites;
      }
      AddLink(s_a, s_b, link_wt);
    }
  }

  // add links to connect the time slices with periodic boundary conditions
  for (int s = 0; s < n_sites; s++) {
    AddLink(s, (s + n_sites_slice) % n_sites, t_scale / cosh(rho[s]));
  }
}

double QfeLatticeAdS3::Sigma(int s1, int s2) {
  if (s1 == s2) return 0.0;

  // this is a relatively slow version but it correctly deals with edge cases
  double rho1 = rho[s1];
  double rho2 = rho[s2];
  double dt = DeltaT(s1, s2);
  double theta = Theta(s1, s2);
  double x1 = cosh(dt) * cosh(rho1) * cosh(rho2);
  double x2 = cos(theta) * sinh(rho1) * sinh(rho2);
  return acosh(x1 - x2);
}

double QfeLatticeAdS3::DeltaT(int s1, int s2) {
  int dt = (Nt / 2) - abs(abs(t[s1] - t[s2]) - (Nt / 2));
  return double(dt) / t_scale;
}
