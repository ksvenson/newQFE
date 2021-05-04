// overrelax.h

#pragma once

class QfeLattice;

class QfeOverrelax {

public:
  void Init(QfeLattice* lattice);
  double Update();
  int UpdateSite(int s);

  QfeLattice* lattice;
  double demon;
};
