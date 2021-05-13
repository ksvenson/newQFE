// phi4.h

#pragma once

#include <vector>

class QfeLattice;

class QfePhi4 {
public:
  QfePhi4(QfeLattice* lattice, double musq, double lambda);
  double Action();
  double MeanPhi();
  void HotStart();
  void ColdStart();
  double Metropolis();
  double Overrelax();
  int WolffUpdate();

  QfeLattice* lattice;
  std::vector<double> phi;  // scalar field
  double lambda;  // bare coupling
  double musq;  // bare mass squared

  double metropolis_z;
  double overrelax_demon;
  std::vector<bool> is_clustered;  // keeps track of which sites are clustered
  std::vector<int> wolff_cluster;  // array of clustered sites
};
