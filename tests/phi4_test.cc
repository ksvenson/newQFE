// lattice_test.cc

#include <cmath>
#include <cstdio>
#include <vector>

#include "phi4.h"
#include "statistics.h"

// phi4 theory on a square lattice has a critical point near lambda = 0.25,
// msq = -1.27251 [1]. We perform a combination of Wolff cluster updates,
// Metropolis updates, overrelaxation updates and Swendsen-Wang updates such
// that we obtain an independent sample every 20 sweeps. We take measurements
// every 20 iterations, which leads to reasonable autocorrelation times for the
// 2nd and 4th magnetic moments. With these parameters, we expect the 4th order
// binder cumulant to be near its critical value of ~0.9. we also expect to see
// a peak in the magnetic susceptibility, which can be seen by varying msq. We
// can also check that the overralation demon is close to 1.

// [1] D. Schaich, W. Loinaz, Phys. Rev. D 79, 056008 (2009).

int main(int argc, char* argv[]) {
  int N = 64;
  printf("N: %d\n", N);

  double msq = -1.27251;
  printf("msq: %.6f\n", msq);

  double lambda = 0.25;
  printf("lambda: %.4f\n", lambda);

  QfeLattice lattice;
  lattice.InitRect(N, N, 1.0, 1.0);
  double vol = lattice.vol;

  QfePhi4 field(&lattice, msq, lambda);
  field.HotStart();
  field.metropolis_z = 2.0;

  printf("initial action: %.12f\n", field.Action());

  // measurements
  QfeMeasReal mag;    // magnetization
  QfeMeasReal mag_2;  // magnetization^2
  QfeMeasReal mag_4;  // magnetization^4
  QfeMeasReal action;
  QfeMeasReal demon;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;
  QfeMeasReal accept_overrelax;

  int n_therm = 5000;
  int n_traj = 20000;
  int n_skip = 20;
  int n_metropolis = 8;
  int n_wolff = 5;
  for (int n = 0; n < (n_traj + n_therm); n++) {
    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      cluster_size_sum += field.WolffUpdate();
    }
    double metropolis_sum = 0.0;
    for (int j = 0; j < n_metropolis; j++) {
      metropolis_sum += field.Metropolis();
    }
    cluster_size.Measure(double(cluster_size_sum) / vol);
    accept_metropolis.Measure(metropolis_sum);
    accept_overrelax.Measure(field.Overrelax());
    demon.Measure(field.overrelax_demon);

    if (n % n_skip || n < n_therm) continue;
    field.SWUpdate();

    action.Measure(field.Action());
    double m = field.MeanPhi();
    double m_sq = m * m;
    mag.Measure(fabs(m));
    mag_2.Measure(m_sq);
    mag_4.Measure(m_sq * m_sq);
    printf("%06d %.12f %+.12f %.4f %.4f %.12f %.4f\n", n, action.last, mag.last,
           accept_metropolis.last, accept_overrelax.last, demon.last,
           cluster_size.last);
  }

  double m_mean = mag.Mean();
  double m_err = mag.Error();
  double m2_mean = mag_2.Mean();
  double m2_err = mag_2.Error();
  double m4_mean = mag_4.Mean();
  double m4_err = mag_4.Error();

  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());
  printf("accept_overrelax: %.4f\n", accept_overrelax.Mean());
  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("demon: %.12f (%.12f)\n", demon.Mean(), demon.Error());
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

  double m_susc_mean = (m2_mean - m_mean * m_mean) * vol;
  double m_susc_err =
      sqrt(pow(m2_err, 2.0) + pow(2.0 * m_mean * m_err, 2.0)) * vol;
  printf("m_susc: %.12e %.12e\n", m_susc_mean, m_susc_err);

  return 0;
}
