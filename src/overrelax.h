// overrelax.h

#pragma once

class QfeLattice;

class QfeOverrelax {

public:
  QfeOverrelax(QfeLattice* lattice);
  double Update();
  int UpdateSite(int s);

  QfeLattice* lattice;
  double demon;
};
