// ads2.cc

#include "ads2.h"

#include <cmath>

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

QfeLatticeAdS2::QfeLatticeAdS2(int n_levels, int q) {
  this->n_levels = n_levels;
  this->q = q;

  double link_wt = 1.0;
  double site_wt = 1.0;

  // create site 0 at the origin
  sites.resize(1);
  sites[0].wt = site_wt;
  sites[0].nn = 0;

  // keep track of level size and offset of first site in each level
  // we need one extra level of dummy sites for dirichlet boundary conditions
  level_size.resize(n_levels + 2);
  level_offset.resize(n_levels + 2);
  level_size[0] = 1;
  level_offset[0] = 0;

  z.resize(1, 0.0);
  const double sin_q = sin(M_PI / double(q));
  const double cos_q = cos(M_PI / double(q));
  const Complex w = 2.0 * I * sin_q;

  for (int n = 1; n <= (n_levels + 1); n++) {

    // determine the number of sites in this level
    if (n == 1) {
      level_size[n] = q;
    } else if (n == 2) {
      level_size[n] = q * (q - 4);
    } else {
      // the recursion relation is only valid for n > 2
      level_size[n] = (q - 4) * level_size[n - 1] - level_size[n - 2];
    }

    // determine the offset of the first site in this level
    level_offset[n] = level_offset[n - 1] + level_size[n - 1];

    // resize the array of sites
    int new_size = sites.size() + level_size[n];
    sites.resize(new_size);
    z.resize(new_size);
    if (n <= n_levels) {
      // don't include dummy sites in n_sites
      n_sites = new_size;
    }

    // add sites to fill up the level
    int p = level_offset[n - 1];  // previous level site to connect to
    for (int c = 0; c < level_size[n]; c++) {

      // init a site and connect it to the previous level
      int s = level_offset[n] + c;
      sites[s].wt = site_wt;
      sites[s].nn = 0;
      AddLink(p, s, link_wt);

      if (s == 1) {
        // first site in level 1 is on the positive real axis
        z[s] = sqrt(1.0 - 4.0 * sin_q * sin_q);
        continue;
      }

      // do some magic to find the complex coordinate of the new site
      double norm_z = norm(z[p]);
      Complex a(cos_q * (1.0 - norm_z), sin_q * (1.0 + norm_z));
      Complex w1 = w * z[p];
      z[s] = (w1 - a * z[s - 1]) / (conj(w1) * z[s - 1] - conj(a));

      // check if site on previous level is full (skip if on 1st level)
      if (sites[p].nn != q || n == 1) continue;

      // go to next site in previous level
      p++;

      // add a link to the current site to make a triangle
      AddLink(p, s, link_wt);
    }

    // connect last site of previous level to first site of current level
    if (n != 1) {
      AddLink(p, level_offset[n], link_wt);
    }

    // exit the loop if we're at the dummy level
    // dummy sites aren't connected to each other
    if (n == (n_levels + 1)) break;

    // connect current level sites to one another in a circle
    for (int c = 0; c < level_size[n]; c++) {
      int s = c + level_offset[n];
      int sp1 = (c + 1) % level_size[n] + level_offset[n];
      AddLink(s, sp1, link_wt);
    }
  }

  // identify the number of sites of each type (bulk, boundary, dummy)
  n_bulk = level_offset[n_levels];
  n_boundary = level_size[n_levels];
  n_dummy = level_size[n_levels + 1];

  // calculate site coordinates in various forms (including dummy sites)
  r.resize(z.size());
  theta.resize(z.size());
  rho.resize(z.size());
  u.resize(z.size());
  for (int s = 0; s < z.size(); s++) {
    r[s] = abs(z[s]);
    theta[s] = arg(z[s]);
    rho[s] = log((1 + r[s]) / (1 - r[s]));
    u[s] = (z[s] + I) / (1.0 + I * z[s]);
  }
}

/**
 * @brief Compute the geodesic distance between lattice sites s1 and s2.
 */

double QfeLatticeAdS2::Sigma(int s1, int s2) {
  Complex z1 = z[s1];
  Complex z2 = z[s2];
  double a = abs(1.0 - conj(z1) * z2);
  double b = abs(z1 - z2);
  return log((a + b) / (a - b));
}

/**
 * @brief Compute the angular distance between two lattice sites.
 * Returns a value between 0 and pi.
 */

double QfeLatticeAdS2::Theta(int s1, int s2) {
  return M_PI - abs(abs(theta[s1] - theta[s2]) - M_PI);
}
