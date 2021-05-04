// metropolis.h

#pragma once

class QfeLattice;

class QfeMetropolis {

public:
  void Init(QfeLattice* lattice);
  double Update();
  double UpdateSite(int s);

  QfeLattice* lattice;
  double z;
};
