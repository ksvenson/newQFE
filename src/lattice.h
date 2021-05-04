// lattice.h

#pragma once

#include <vector>
#include <random>
#include "rng.h"

#define MAX_NEIGHBORS 6

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
  double Action();
  void HotStart();
  void ColdStart();

  inline int n_sites() { return sites.size(); }
  inline int n_links() { return links.size(); }

  // general qfe lattice variables
  std::vector<QfeSite> sites;
  std::vector<QfeLink> links;

  QfeRng rng;

  // phi^4 specific variables
  std::vector<double> phi;  // phi field
  double mag;  // magnetization per site (sum_x phi_x / n_sites)
  double lambda;  // bare coupling
  double musq;  // bare mass squared
};
