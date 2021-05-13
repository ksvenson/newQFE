// ising.h

#pragma once

#include <vector>

class QfeLattice;

class QfeIsing {
public:
  QfeIsing(QfeLattice* lattice, double beta);
  double Action();
  double MeanSpin();
  void HotStart();
  void ColdStart();
  double Metropolis();
  int WolffUpdate();

  QfeLattice* lattice;
  std::vector<double> spin;  // Z2 field
  double beta;  // bare coupling

  std::vector<bool> is_clustered;  // keeps track of which sites are clustered
  std::vector<int> wolff_cluster;  // array of clustered sites
};
