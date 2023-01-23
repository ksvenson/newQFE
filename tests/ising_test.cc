// ising_test.cc

#include "ising.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "statistics.h"

// For the ising model on a equilateral triangular lattice with no external
// field, we expect to find a critical point at beta = ln(3)/4. Compare with
// [1] which finds T_c = 1 / beta_c ~ 3.64. We perform 2 wolff updates and 3
// metropolis updates, which for a 64^2 lattice leads a net wolff cluster
// roughly equal to the lattice size and a net metropolis acceptance of about 1.
// With these parameters, we expect the 4th order binder cumulant to be near its
// critical value of ~0.9. we also expect to see a peak in the magnetic
// susceptibility, which can be seen by varying beta.

// [1] Z. Luo, et al., Chinese Phys. B 18, 2696 (2009).

int main(int argc, char* argv[]) {
  int N = 64;
  printf("N: %d\n", N);

  double skew = 0.0;
  printf("skew: %.2f\n", skew);

  double beta = 0.25 * log(3.0);
  printf("beta: %.4f\n", beta);

  QfeLattice lattice;
  lattice.InitTriangle(N, 1.0, 1.0, 1.0);

  QfeIsing field(&lattice, beta);
  field.HotStart();

  printf("initial action: %.12f\n", field.Action());

  // measurements
  QfeMeasReal mag;
  QfeMeasReal mag_2;
  QfeMeasReal mag_4;
  QfeMeasReal action;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;

  int n_therm = 5000;
  int n_traj = 20000;
  int n_skip = 20;
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
    cluster_size.Measure(double(cluster_size_sum) / lattice.vol);
    accept_metropolis.Measure(metropolis_sum);

    if (n % n_skip || n < n_therm) continue;
    field.SWUpdate();

    action.Measure(field.Action());
    double m = fabs(field.MeanSpin());
    double m2 = m * m;
    mag.Measure(m);
    mag_2.Measure(m2);
    mag_4.Measure(m2 * m2);
    printf("%06d %.12f %+.12f %.4f %.4f\n", n, action.last, mag.last,
           accept_metropolis.last, cluster_size.last);
  }

  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());
  printf("cluster_size/V: %.4f\n", cluster_size.Mean());

  double m_mean = mag.Mean();
  double m_err = mag.Error();
  double m2_mean = mag_2.Mean();
  double m2_err = mag_2.Error();
  double m4_mean = mag_4.Mean();
  double m4_err = mag_4.Error();

  printf("action: %+.12e %.12e %.4f %.4f\n", action.Mean(), action.Error(),
         action.AutocorrFront(), action.AutocorrBack());
  printf("mag: %.12e %.12e %.4f %.4f\n", m_mean, m_err, mag.AutocorrFront(),
         mag.AutocorrBack());
  printf("m^2: %.12e %.12e %.4f %.4f\n", m2_mean, m2_err, mag_2.AutocorrFront(),
         mag_2.AutocorrBack());
  printf("m^4: %.12e %.12e %.4f %.4f\n", m4_mean, m4_err, mag_4.AutocorrFront(),
         mag_4.AutocorrBack());

  double U4_mean = 1.5 * (1.0 - m4_mean / (3.0 * m2_mean * m2_mean));
  double U4_err =
      0.5 * U4_mean *
      sqrt(pow(m4_err / m4_mean, 2.0) + pow(2.0 * m2_err / m2_mean, 2.0));
  printf("U4: %.12e %.12e\n", U4_mean, U4_err);

  double m_susc_mean = (m2_mean - m_mean * m_mean) * lattice.vol;
  double m_susc_err =
      sqrt(pow(m2_err, 2.0) + pow(2.0 * m_mean * m_err, 2.0)) * lattice.vol;
  printf("m_susc: %.12e %.12e\n", m_susc_mean, m_susc_err);

  return 0;
}
