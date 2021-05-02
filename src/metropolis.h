// metropolis.h

#pragma once

class QfeLattice;

class QfeMetropolis {

 public:
  void Init(QfeLattice* lattice);
  double Update();

 protected:
  double UpdateSite(int s);

 private:
  QfeLattice* lattice;
  double z;
};
