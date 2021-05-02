// overrelax.h

#pragma once

class QfeLattice;

class QfeOverrelax {

 public:
  void Init(QfeLattice* lattice);
  double Update();
  inline double GetDemon() { return demon; }

 protected:
  int UpdateSite(int s);

 private:
   QfeLattice* lattice;
   double demon;
};
