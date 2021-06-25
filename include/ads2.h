// ads2.h

#pragma once

#include <cmath>
#include <complex>
#include <vector>
#include "lattice.h"

typedef std::complex<double> Complex;
const Complex I(0.0, 1.0);

class QfeLatticeAdS2 : public QfeLattice {

public:
  QfeLatticeAdS2(int n_layers, int q);
  double Sigma(int s1, int s2);
  double Theta(int s1, int s2);

  int q;  // number of vertices meeting at each lattice point
  int n_layers;  // number of layers
  int n_bulk;  // number of bulk sites
  int n_boundary;  // number of boundary sites
  std::vector<int> layer_size;  // size of each layer
  std::vector<int> layer_offset;  // offset of first site in each layer

  std::vector<std::vector<int>> layer_sites;  // list of sites at each layer
  std::vector<int> bulk_sites;  // list of bulk sites
  std::vector<int> boundary_sites;  // list of boundary sites
  std::vector<int> site_layers;  // layer for each site

  // site coordinates
  std::vector<Complex> z;  // complex coordinates on poincaré disc
  std::vector<double> r;  // abs(z)
  std::vector<double> theta;  // arg(z)
  std::vector<double> rho;  // global radial coordinate
  std::vector<Complex> u;  // complex coordinates in upper half-plane

  std::vector<double> layer_rho;  // average rho at each layer
  std::vector<double> layer_cosh_rho;  // average cosh(rho) at each layer
  std::vector<double> total_cosh_rho;  // average cosh(rho) up to each layer
};

/**
 * @brief Initialize a triangulated lattice on AdS2.
 *
 * The first point is placed at the origin of a Poincaré disk. The number of
 * nearest neighbors is @p q, which must be greater than 6 to get negative
 * curvature. The first layer has @p q sites in a circle around the origin.
 * Additional layers are added according to the procedure in [1] until the
 * number of layers equals @p n_layers. All sites and links have equal weight.
 *
 * [1] R. Brower et al., Phys. Rev. D, 103, 094507 (2021).
 * @see https://arxiv.org/abs/1912.07606
 *
 * @param n_layers Number of layers to create
 * @param q Number of triangles meeting at each site (should be greater than 6)
 */

QfeLatticeAdS2::QfeLatticeAdS2(int n_layers, int q) {
  this->n_layers = n_layers;
  this->q = q;

  double link_wt = 1.0;
  double site_wt = 1.0;

  // create site 0 at the origin
  sites.resize(1);
  site_layers.resize(1);
  sites[0].wt = site_wt;
  sites[0].nn = 0;
  site_layers[0] = 0;

  // keep track of layer size and offset of first site in each layer
  // we need one extra layer of dummy sites for dirichlet boundary conditions
  layer_size.resize(n_layers + 2);
  layer_offset.resize(n_layers + 2);
  layer_sites.resize(n_layers + 2);
  layer_size[0] = 1;
  layer_offset[0] = 0;
  layer_sites[0].push_back(0);
  bulk_sites.push_back(0);

  z.resize(1, 0.0);
  const double sin_q = sin(M_PI / double(q));
  const double cos_q = cos(M_PI / double(q));
  const Complex w = 2.0 * I * sin_q;

  for (int n = 1; n <= (n_layers + 1); n++) {

    // determine the number of sites in this layer
    if (n == 1) {
      layer_size[n] = q;
    } else if (n == 2) {
      layer_size[n] = q * (q - 4);
    } else {
      // the recursion relation is only valid for n > 2
      layer_size[n] = (q - 4) * layer_size[n - 1] - layer_size[n - 2];
    }

    // determine the offset of the first site in this layer
    layer_offset[n] = layer_offset[n - 1] + layer_size[n - 1];

    // resize the array of sites
    int new_size = sites.size() + layer_size[n];
    sites.resize(new_size);
    z.resize(new_size);
    site_layers.resize(new_size);
    if (n <= n_layers) {
      // don't include dummy sites in n_sites
      n_sites = new_size;
    }

    // add sites to fill up the layer
    int p = layer_offset[n - 1];  // previous layer site to connect to
    for (int c = 0; c < layer_size[n]; c++) {

      // init a site and connect it to the previous layer
      int s = layer_offset[n] + c;
      sites[s].wt = site_wt;
      sites[s].nn = 0;
      layer_sites[n].push_back(s);
      if (n < n_layers) {
        bulk_sites.push_back(s);
      } else if (n == n_layers) {
        boundary_sites.push_back(s);
      }
      site_layers[s] = n;
      AddLink(p, s, link_wt);

      if (s == 1) {
        // first site in layer 1 is on the positive real axis
        z[s] = sqrt(1.0 - 4.0 * sin_q * sin_q);
        continue;
      }

      // do some magic to find the complex coordinate of the new site
      double norm_z = norm(z[p]);
      Complex a(cos_q * (1.0 - norm_z), sin_q * (1.0 + norm_z));
      Complex w1 = w * z[p];
      z[s] = (w1 - a * z[s - 1]) / (conj(w1) * z[s - 1] - conj(a));

      // check if site on previous layer is full (skip if on 1st layer)
      if (sites[p].nn != q || n == 1) continue;

      // go to next site in previous layer
      p++;

      // add a link to the current site to make a triangle
      AddLink(p, s, link_wt);
    }

    // connect last site of previous layer to first site of current layer
    if (n != 1) {
      AddLink(p, layer_offset[n], link_wt);
    }

    // exit the loop if we're at the dummy layer
    // dummy sites aren't connected to each other
    if (n == (n_layers + 1)) break;

    // connect current layer sites to one another in a circle
    for (int c = 0; c < layer_size[n]; c++) {
      int s = c + layer_offset[n];
      int sp1 = (c + 1) % layer_size[n] + layer_offset[n];
      AddLink(s, sp1, link_wt);
    }
  }

  // identify the number of sites of each type (bulk, boundary, dummy)
  n_bulk = layer_offset[n_layers];
  n_boundary = layer_size[n_layers];
  n_dummy = layer_size[n_layers + 1];

  // calculate site coordinates in various forms (including dummy sites)
  r.resize(z.size());
  theta.resize(z.size());
  rho.resize(z.size());
  u.resize(z.size());
  for (int s = 0; s < z.size(); s++) {
    r[s] = abs(z[s]);
    theta[s] = arg(z[s]);
    rho[s] = log((1 + r[s]) / (1 - r[s]));
    u[s] = I * (1.0 + z[s]) / (1.0 - z[s]);
  }

  // update average rho on each layer
  layer_rho.resize(n_layers + 2);
  layer_cosh_rho.resize(n_layers + 2);
  total_cosh_rho.resize(n_layers + 2);
  double total_cosh_rho_sum = 0.0;
  for (int n = 0; n <= (n_layers + 1); n++) {
    double rho_sum = 0.0;
    double cosh_rho_sum = 0.0;
    for (int i = 0; i < layer_size[n]; i++) {
      int s = layer_offset[n] + i;
      rho_sum += rho[s];
      cosh_rho_sum += cosh(rho[s]);
      total_cosh_rho_sum += cosh(rho[s]);
    }
    layer_rho[n] = rho_sum / double(layer_size[n]);
    layer_cosh_rho[n] = cosh_rho_sum / double(layer_size[n]);
    total_cosh_rho[n] = total_cosh_rho_sum / double(layer_offset[n] + layer_size[n]);
  }
}

/**
 * @brief Compute the geodesic distance between lattice sites s1 and s2.
 */

double QfeLatticeAdS2::Sigma(int s1, int s2) {

  if (s1 == s2) return 0.0;
  const Complex z1 = z[s1];
  const Complex z2 = z[s2];
  const double a = abs(1.0 - conj(z1) * z2);
  const double b = abs(z1 - z2);
  return log((a + b) / (a - b));
}

/**
 * @brief Compute the angular distance between two lattice sites.
 * Returns a value between 0 and pi.
 */

double QfeLatticeAdS2::Theta(int s1, int s2) {
  return M_PI - abs(abs(theta[s1] - theta[s2]) - M_PI);
}
