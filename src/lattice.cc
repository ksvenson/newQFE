// lattice.cc

#include "lattice.h"

#include <vector>

// skew is a number between 0 and 1 which determines how skewed the triangles
// are. a value of 0 corresponds to equilateral triangles, and a value of 1
// corresponds to right triangles, which is equivalent to a square lattice
// because the diagonal link weights will be zero.

void QfeLattice::InitTriangle(int N, double skew) {

  // create sites
  sites.resize(N * N);

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

// Creates a triangulated lattice on AdS2. The first point is placed at the
// origin of a Poincaré disk. The number of nearest neighbors is q, which must
// be greater than 6 to get negative curvature. The first layer has q sites
// in a circle around the origin. Additional layers are added according to the
// procedure in [1] until the number of layers equals n_layers. All sites and
// links have equal weight.

// [1] R. Brower et al., Phys. Rev. D, forthcoming (2021). arXiv:1912.07606

void QfeLattice::InitAdS2(int n_layers, int q) {

  double link_wt = 1.0;
  double site_wt = 1.0;

  // create site 0 at the origin
  sites.resize(1);
  sites[0].wt = site_wt;
  sites[0].nn = 0;

  // keep track of layer size and offset of first site in each layer
  int layer_size[n_layers + 1];
  int layer_offset[n_layers + 1];
  layer_size[0] = 1;
  layer_offset[0] = 0;

  for (int n = 1; n <= n_layers; n++) {

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
    sites.resize(n_sites() + layer_size[n]);

    int p = layer_offset[n - 1];  // current site in previous layer
    for (int c = 0; c < layer_size[n]; c++) {

      // init a site and connect it to the previous layer
      int s = layer_offset[n] + c;
      sites[s].wt = site_wt;
      sites[s].nn = 0;
      AddLink(p, s, link_wt);

      // check site on previous layer is full (doesn't apply to 1st layer)
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

    // connect current layer sites to one another in a circle
    for (int c = 0; c < layer_size[n]; c++) {
      int s = c + layer_offset[n];
      int sp1 = (c + 1) % layer_size[n] + layer_offset[n];
      AddLink(s, sp1, link_wt);
    }
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
