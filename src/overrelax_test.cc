// overrelax_test.cc

#include <stdio.h>
#include "lattice.h"
#include "overrelax.h"
#include "metropolis.h"

// an overrelax sweep should leave the quantity (S_quadratic + demon) invariant
// where S_quadratic is the quadratic part of the action and demon is the demon
// parameter. in addition, the expectation value of the demon parameter should
// be 1. this test checks whether both of these conditions are met. because the
// overrelaxation algorithm by itself is not stochastic, a metropolis update
// is performed prior to each overrelaxation update.

int main(int argc, char* argv[]) {

  int N = 64;

  QfeLattice lattice;
  lattice.InitTriangle(N);
  lattice.HotStart();

  QfeMetropolis metropolis;
  metropolis.Init(&lattice);

  QfeOverrelax overrelax;
  overrelax.Init(&lattice);

  double demon_sum = 0.0;
  double demon2_sum = 0.0;
  double delta_S_sum = 0.0;
  double delta_S2_sum = 0.0;

  int n_update = 20000;
  for (int i = 0; i < n_update; i++) {
    metropolis.Update();
    double demon = overrelax.demon;
    double S_before = lattice.Action() + demon / double(lattice.n_sites());
    double accept_rate = overrelax.Update();
    demon = overrelax.demon;
    double S_after = lattice.Action() + demon / double(lattice.n_sites());
    double delta_S = S_after - S_before;
    demon_sum += demon;
    demon2_sum += demon * demon;
    delta_S_sum += delta_S;
    delta_S2_sum += delta_S * delta_S;
    printf("%06d %.4f %.12f %.12e\n", i, accept_rate, demon, delta_S);
  }

  double demon_ave = demon_sum / double(n_update);
  double demon2_ave = demon2_sum / double(n_update);
  double demon_var = demon2_ave - demon_ave * demon_ave;

  double delta_S_ave = delta_S_sum / double(n_update);
  double delta_S2_ave = delta_S2_sum / double(n_update);
  double delta_S_var = delta_S2_ave - delta_S_ave * delta_S_ave;

  printf("<demon> = %.12f (%.12f)\n", demon_ave, sqrt(demon_var / double(n_update)));
  printf("<delta_S> = %.12e (%.12e)\n", delta_S_ave, sqrt(delta_S_var / double(n_update)));

  return 0;
}
