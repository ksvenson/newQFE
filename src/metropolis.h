// metropolis.h

#pragma once

class QfeLattice;

class QfeMetropolis {

public:
  QfeMetropolis(QfeLattice* lattice);
  double Update();
  double UpdateSite(int s);

  QfeLattice* lattice;
  double z;
};
