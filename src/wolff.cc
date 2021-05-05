// wolff.cc

// wolff cluster update algorithm
// ref: U. Wolff, Phys. Rev. Lett. 62, 361 (1989).

#include "wolff.h"

#include <stack>
#include "lattice.h"

void QfeWolff::Init(QfeLattice* lattice) {
  this->lattice = lattice;
  is_clustered.resize(lattice->n_sites());
}

int QfeWolff::Update() {

  // remove all sites from the cluster
  std::fill(is_clustered.begin(), is_clustered.end(), false);
  cluster.clear();

  // create the stack
  std::stack<int> stack;

  // choose a random site and add it to the cluster
  int s = lattice->rng.RandInt(0, lattice->n_sites() - 1);
  AddToCluster(s);
  stack.push(s);

  while (stack.size() != 0) {
    s = stack.top();
    stack.pop();

    // try to add neighbors
    QfeSite* site = &lattice->sites[s];
    double value = lattice->phi[s];
    for (int n = 0; n < site->nn; n++) {
      double link_wt = lattice->links[site->links[n]].wt;
      s = site->neighbors[n];

      // skip if the site is already clustered
      if (is_clustered[s]) continue;

      if (TestSite(s, value * link_wt)) {
        // add the site to the cluster
        AddToCluster(s);
        stack.push(s);
      }
    }
  }

  FlipCluster();

  return cluster.size();
}

bool QfeWolff::TestSite(int s, double test_value) {

  // fail if sign bits don't match
  double value = lattice->phi[s];
  if (signbit(test_value) != signbit(value)) return false;

  if (lattice->rng.RandReal() < (1 - exp(-2 * test_value * value))) {
    // add the site to the cluster
    return true;
  } else {
    return false;
  }
}

void QfeWolff::AddToCluster(int s) {
  cluster.push_back(s);
  is_clustered[s] = true;
}

void QfeWolff::FlipCluster() {

  for (int s = 0; s < cluster.size(); s++) {
    lattice->phi[cluster[s]] *= -1.0;
  }
}
