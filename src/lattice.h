// lattice.h

#pragma once

#include <vector>
#include <random>
#include "rng.h"

#define MAX_NEIGHBORS 12

struct QfeSite {
  double wt;  // site weight
  int nn;  // number of nearest neighbors
  int links[MAX_NEIGHBORS];  // nearest neighbor links
  int neighbors[MAX_NEIGHBORS];  // nearest neighbor sites
};

struct QfeLink {
  double wt;  // link weight
  int sites[2];  // sites attached by this link
};

class QfeLattice {

public:
  void InitTriangle(int N, double skew = 0.0);
  QfeLink AddLink(int a, int b, double wt);

  int n_sites;
  int n_links;

  std::vector<QfeSite> sites;
  std::vector<QfeLink> links;

  QfeRng rng;
};
