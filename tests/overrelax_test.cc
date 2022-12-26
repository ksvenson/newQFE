// overrelax_test.cc

#include <cstdio>

#include "phi4.h"
#include "statistics.h"

// An overrelax sweep should leave the quantity (S_quadratic + demon) invariant
// where S_quadratic is the quadratic part of the action and demon is the demon
// parameter. In addition, the expectation value of the demon parameter should
// be 1. This test checks whether both of these conditions are met. Because the
// overrelaxation algorithm by itself is not ergodic, a metropolis update
// is performed prior to each overrelaxation update.

int main(int argc, char* argv[]) {
  int N = 64;

  QfeLattice lattice;
  lattice.InitRect(N, N, 1.0, 1.0);
  double vol = lattice.vol;

  QfePhi4 field(&lattice, -1.2725, 0.25);
  field.HotStart();

  QfeMeasReal demon;
  QfeMeasReal delta_S;

  int n_update = 20000;
  for (int i = 0; i < n_update; i++) {
    field.Metropolis();
    double demon_old = field.overrelax_demon;
    double S_old = field.Action() + demon_old / vol;
    double accept_rate = field.Overrelax();
    double demon_new = field.overrelax_demon;
    double S_new = field.Action() + demon_new / vol;
    demon.Measure(demon_new);
    delta_S.Measure(S_new - S_old);
    printf("%06d %.4f %.12f %.12e\n", i, accept_rate, demon.last, delta_S.last);
  }

  printf("demon: %.12f (%.12f)\n", demon.Mean(), demon.Error());
  printf("delta_S: %.12e (%.12e)\n", delta_S.Mean(), delta_S.Error());

  return 0;
}
