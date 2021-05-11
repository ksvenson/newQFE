// wolff.h

#pragma once

#include <vector>

class QfeLattice;

class QfeWolff {

public:
  QfeWolff(QfeLattice* lattice);
  int Update();
  bool TestSite(int s, double test_value);
  void AddToCluster(int s);
  void FlipCluster();

  QfeLattice* lattice;
  std::vector<bool> is_clustered;  // keeps track of which sites are clustered
  std::vector<int> cluster;  // array of clustered sites
};
