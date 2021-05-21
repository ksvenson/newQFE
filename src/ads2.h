// ads2.h

#pragma once

#include <vector>
#include <complex>
#include "lattice.h"

typedef std::complex<double> Complex;
const Complex I(0.0, 1.0);

class QfeLatticeAdS2 : public QfeLattice {

public:
  QfeLatticeAdS2(int n_levels, int q);
  double Sigma(int s1, int s2);
  double Theta(int s1, int s2);

  int q;  // number of vertices meeting at each lattice point
  int n_levels;  // number of levels
  int n_bulk;  // number of bulk sites
  int n_boundary;  // number of boundary sites
  std::vector<int> level_size;  // size of each level
  std::vector<int> level_offset;  // offset of first site in each level

  // site coordinates
  std::vector<Complex> z;  // complex coordinates on poincaré disc
  std::vector<double> r;  // abs(z)
  std::vector<double> theta;  // arg(z)
  std::vector<double> rho;  // global radial coordinate
  std::vector<Complex> u;  // complex coordinates in upper half-plane
};
