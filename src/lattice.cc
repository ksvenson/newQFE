// lattice.cc

#include "lattice.h"

#include <vector>

/**
 * @brief Creates a flat triangulated lattice with periodic boundary
 * conditions.
 *
 * @param N Lattice size
 * @param skew A value from 0 and 1 which determines how skewed the triangles
 * are. a value of 0 corresponds to equilateral triangles, and a value of 1
 * corresponds to right triangles, which is equivalent to a square lattice
 * because the diagonal link weights will be zero.
 */

void QfeLattice::InitTriangle(int N, double skew) {

  // create sites
  n_sites = N * N;
  sites.resize(n_sites);

  // set all site weights to 1.0
  for (int s = 0; s < n_sites; s++) {
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

  for (int s = 0; s < n_sites; s++) {
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
  n_links = links.size();
}

/**
 * @brief Adds a link from site @p a to @p b with weight @p wt.
 */

QfeLink QfeLattice::AddLink(int a, int b, double wt) {

  int l = links.size();  // link index
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
