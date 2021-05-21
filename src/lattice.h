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
  QfeLattice();
  void InitTriangle(int N, double skew = 0.0);
  QfeLink AddLink(int a, int b, double wt);
  void PrintSites();
  void PrintLinks();
  void CheckConnectivity();
  void CheckConsistency();

  int n_sites;
  int n_dummy;  // number of dummy sites for dirichlet boundary conditions
  int n_links;

  std::vector<QfeSite> sites;
  std::vector<QfeLink> links;

  QfeRng rng;
};
