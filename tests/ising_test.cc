// ising_test.cc

#include <cstdio>
#include <vector>
#include "lattice.h"
#include "ising.h"
#include "statistics.h"

using std::vector;

// for the ising model on a equilateral triangular lattice with no external
// field, we expect to find a critical point near beta = 0.41. compare with
// [1] which finds T_c = 1 / beta_c ~ 3.5, but note that using our convention
// with finite element link weights, the temperature is different by a factor
// of 2/3. we perform 2 wolff updates and 3 metropolis updates, which for a
// 64^2 lattice leads a net wolff cluster roughly equal to the lattice size
// and a net metropolis acceptance of about 1. with these parameters, we expect
// the 4th order binder cumulant to be near its critical value of 0.8. we also
// expect to see a peak in the magnetic susceptibility, which can be seen by
// varying beta.

// [1] Z. Luo, et al., Chinese Phys. B 18, 2696 (2009).

int main(int argc, char* argv[]) {

  int N = 64;
  printf("N: %d\n", N);

  double skew = 0.0;
  printf("skew: %.2f\n", skew);

  double beta = 0.41;
  printf("beta: %.4f\n", beta);

  QfeLattice lattice;
  lattice.InitTriangle(N, skew);

  QfeIsing field(&lattice, beta);
  field.HotStart();

  printf("initial action: %.12f\n", field.Action());

  // measurements
  vector<double> mag;
  vector<double> action;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;

  int n_therm = 1000;
  int n_traj = 20000;
  int n_skip = 2;
  int n_wolff = 3;
  int n_metropolis = 5;
  for (int n = 0; n < (n_traj + n_therm); n++) {

    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      cluster_size_sum += field.WolffUpdate();
    }
    double metropolis_sum = 0.0;
    for (int j = 0; j < n_metropolis; j++) {
      metropolis_sum += field.Metropolis();
    }
    cluster_size.Measure(double(cluster_size_sum) / double(N * N));
    accept_metropolis.Measure(metropolis_sum);

    if (n % n_skip || n < n_therm) continue;

    action.push_back(field.Action());
    mag.push_back(field.MeanSpin());
    printf("%06d %.12f %+.12f %.4f %.4f\n", \
        n, action.back(), mag.back(), \
        accept_metropolis.last, \
        cluster_size.last);
  }

  vector<double> mag_abs(mag.size());
  vector<double> mag2(mag.size());
  vector<double> mag4(mag.size());
  for (int i = 0; i < mag.size(); i++) {
    double m = mag[i];
    double m2 = m * m;
    mag_abs[i] = abs(m);
    mag2[i] = m2;
    mag4[i] = m2 * m2;
  }

  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());
  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
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
