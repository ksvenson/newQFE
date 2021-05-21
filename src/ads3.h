// ads3.h

#pragma once

#include "ads2.h"

class QfeLatticeAdS3 : public QfeLatticeAdS2 {

public:
  QfeLatticeAdS3(int n_levels, int q, int Nt);
  double Sigma(int s1, int s2);
  double Theta(int s1, int s2);

  int Nt;
  std::vector<int> t;  // time coordinate of sites
};
