// lattice_test.cc

#include <stdio.h>
#include "lattice.h"
#include "wolff.h"
#include "overrelax.h"
#include "metropolis.h"
#include "statistics.h"

int main(int argc, char* argv[]) {

  int N = 64;
  printf("N: %d\n", N);

  double skew = 1.0;
  printf("skew: %.2f\n", skew);

  double musq = 1.4;
  printf("musq: %.4f\n", musq);

  double lambda = 0.25;
  printf("lambda: %.4f\n", lambda);

  QfeLattice lattice(musq, lambda);
  lattice.InitTriangle(N, skew);
  lattice.HotStart();

  QfeMetropolis metropolis(&lattice);
  QfeOverrelax overrelax(&lattice);
  QfeWolff wolff(&lattice);

  printf("Initial Action: %.12f\n", lattice.Action());

  // measurements
  std::vector<double> mag;
  std::vector<double> action;
  std::vector<double> demon;
  std::vector<double> cluster_size;
  std::vector<double> accept_metropolis;
  std::vector<double> accept_overrelax;

  int n_therm = 1000;
  int n_traj = 40000;
  int n_skip = 20;
  int n_wolff = 5;
  for (int n = 0; n < (n_traj + n_therm); n++) {

    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      cluster_size_sum += wolff.Update();
    }
    cluster_size.push_back(double(cluster_size_sum) / double(N * N));
    accept_metropolis.push_back(metropolis.Update());
    accept_overrelax.push_back(overrelax.Update());
    demon.push_back(overrelax.demon);

    if (n % n_skip || n < n_therm) continue;

    action.push_back(lattice.Action());
    mag.push_back(lattice.mag);
    printf("%06d %.12f %+.12f %.4f %.4f %.12f %d\n", \
        n, action.back(), mag.back(), \
        accept_metropolis.back(), \
        accept_overrelax.back(), demon.back(), \
        cluster_size_sum);
  }

  std::vector<double> mag_abs(mag.size());
  std::vector<double> mag2(mag.size());
  std::vector<double> mag4(mag.size());
  for (int i = 0; i < mag.size(); i++) {
    double m = mag[i];
    double m2 = m * m;
    mag_abs[i] = abs(m);
    mag2[i] = m2;
    mag4[i] = m2 * m2;
  }

  printf("accept_metropolis: %.4f\n", Mean(accept_metropolis));
  printf("accept_overrelax: %.4f\n", Mean(accept_overrelax));
  printf("cluster_size/V: %.4f\n", Mean(cluster_size));
  printf("demon: %.12f (%.12f)\n", Mean(demon), JackknifeMean(demon));
  printf("action: %.12e (%.12e), %.4f\n", \
      Mean(action), JackknifeMean(action), AutocorrTime(action));
  printf("m: %.12e (%.12e), %.4f\n", \
      Mean(mag), JackknifeMean(mag), AutocorrTime(mag));
  printf("m^2: %.12e (%.12e), %.4f\n", \
      Mean(mag2), JackknifeMean(mag2), AutocorrTime(mag2));
  printf("m^4: %.12e (%.12e), %.4f\n", \
      Mean(mag4), JackknifeMean(mag4), AutocorrTime(mag4));
  printf("U4: %.12e (%.12e)\n", U4(mag2, mag4), JackknifeU4(mag2, mag4));
  printf("susceptibility: %.12e (%.12e)\n", Susceptibility(mag2, mag_abs), \
      JackknifeSusceptibility(mag2, mag_abs));

  return 0;
}
