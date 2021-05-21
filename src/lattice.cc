// lattice.cc

#include "lattice.h"

#include <vector>
#include <stack>

QfeLattice::QfeLattice() {
  n_sites = 0;
  n_dummy = 0;
  n_links = 0;
}

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
}

/**
 * @brief Adds a link from site @p a to @p b with weight @p wt.
 */

QfeLink QfeLattice::AddLink(int a, int b, double wt) {

  int l = links.size();  // link index
  QfeLink link;
  link.wt = wt;
  link.sites[0] = a;
  link.sites[1] = b;

  // add site neighbors only if sites are dynamic
  if (a < n_sites) {
    int nn_a = sites[a].nn;
    sites[a].neighbors[nn_a] = b;
    sites[a].links[nn_a] = l;
    sites[a].nn++;
  }

  if (b < n_sites) {
    int nn_b = sites[b].nn;
    sites[b].neighbors[nn_b] = a;
    sites[b].links[nn_b] = l;
    sites[b].nn++;
  }

  links.push_back(link);
  n_links = links.size();
  return link;
}

/**
 * @brief Print a list of sites with their weights and neighbors
 */

void QfeLattice::PrintSites() {
  for (int s = 0; s < sites.size(); s++) {
    printf("%04d", s);
    printf(" %.12f", sites[s].wt);
    for (int n = 0; n < sites[s].nn; n++) {
      printf(" %04d", sites[s].neighbors[n]);
    }
    printf("\n");
  }
}

/**
 * @brief Print a list of links with their weights and attached sites
 */

void QfeLattice::PrintLinks() {
  for (int l = 0; l < links.size(); l++) {
    printf("%04d", l);
    printf(" %.12f", links[l].wt);
    printf(" %04d", links[l].sites[0]);
    printf(" %04d", links[l].sites[1]);
    printf("\n");
  }
}

/**
 * @brief Check that all lattice sites are connected.
 */

void QfeLattice::CheckConnectivity() {

  printf("\n*** connectivity check ***\n");
  printf("dynamic sites: %d\n", n_sites);
  printf("dummy sites: %d\n", int(sites.size()) - n_sites);
  printf("total sites: %d\n", int(sites.size()));

  // keep track of which sites are connected (include dummy sites)
  std::vector<bool> is_connected(sites.size());

  // create the stack
  std::stack<int> stack;

  // start with site 0
  stack.push(0);
  is_connected[0] = true;

  int n_connected = 0;

  while (stack.size() != 0) {
    n_connected++;
    int s = stack.top();
    stack.pop();
    QfeSite* site = &sites[s];

    for (int n = 0; n < site->nn; n++) {
      int s = site->neighbors[n];
      if (is_connected[s]) continue;
      is_connected[s] = true;
      stack.push(s);
    }
  }

  printf("connected sites: %d\n", n_connected);
  printf("disconnected sites: %d\n", int(sites.size()) - n_connected);
}

/**
 * @brief Check that the neighbor lists match the link sites.
 */

void QfeLattice::CheckConsistency() {

  printf("\n*** consistency check ***\n");

  // make sure each site's neighbor table is consistent with its links
  for (int l = 0; l < links.size(); l++) {

    int s_a = links[l].sites[0];
    int s_b = links[l].sites[1];

    // check 1st site
    if (s_a < n_sites) {  // skip dummy sites
      int n;
      for (n = 0; n < sites[s_a].nn; n++) {
        if (sites[s_a].links[n] == l) break;
      }

      if (n == sites[s_a].nn) {
        // link not found
        printf("link %04d not found in neighbor table for site %04d\n", l, s_a);
      } else if (sites[s_a].neighbors[n] != s_b) {
        printf("site %04d neighbor %04d mismatch (link %04d, site %04d)\n", \
            s_a, sites[s_a].neighbors[n], l, s_b);
      }
    }

    // check 2nd site
    if (s_b < n_sites) {  // skip dummy sites
      int n;
      for (n = 0; n < sites[s_b].nn; n++) {
        if (sites[s_b].links[n] == l) break;
      }

      if (n == sites[s_b].nn) {
        // link not found
        printf("link %04d not found in neighbor table for site %04d\n", l, s_b);
      } else if (sites[s_b].neighbors[n] != s_a) {
        printf("site %04d neighbor %04d mismatch (link %04d, site %04d)\n", \
            s_b, sites[s_b].neighbors[n], l, s_a);
      }
    }
  }
}
