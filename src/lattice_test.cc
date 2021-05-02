// lattice_test.cc

#include <stdio.h>
#include "lattice.h"
#include "wolff.h"
#include "overrelax.h"
#include "metropolis.h"

int main(int argc, char* argv[]) {

  int N = 48;
  printf("N: %d\n", N);

  QfeLattice lattice;
  lattice.InitTriangle(N, 1.0);
  lattice.HotStart();

  QfeMetropolis metropolis;
  metropolis.Init(&lattice);

  QfeOverrelax overrelax;
  overrelax.Init(&lattice);

  QfeWolff wolff;
  wolff.Init(&lattice);

  printf("Initial Action: %.12f\n", lattice.Action());

  double mag_sum = 0.0;
  double mag2_sum = 0.0;
  double mag4_sum = 0.0;
  double mag8_sum = 0.0;
  int cluster_sum = 0;
  int n_update = 20000;
  for (int i = 0; i < n_update; i++) {

    int cluster_size = 0;
    for (int j = 0; j < 5; j++) {
      cluster_size += wolff.Update();
    }
    double accept_metropolis = metropolis.Update();
    double accept_overrelax = overrelax.Update();

    double mag = lattice.GetMag();
    double mag2 = mag * mag;
    double mag4 = mag2 * mag2;
    mag_sum += mag;
    mag2_sum += mag2;
    mag4_sum += mag2 * mag2;
    mag8_sum += mag4 * mag4;
    cluster_sum += cluster_size;
    printf("%06d %.12f %.12f %.4f %.4f %.12f %d\n", i, \
        lattice.Action(), lattice.GetMag(), \
        accept_metropolis, \
        accept_overrelax, overrelax.GetDemon(), \
        cluster_size);
  }

  double cluster_ave = double(cluster_sum) / double(n_update * lattice.n_sites());

  double m_ave = mag_sum / double(n_update);
  double m2_ave = mag2_sum / double(n_update);
  double m4_ave = mag4_sum / double(n_update);
  double m8_ave = mag8_sum / double(n_update);

  double m_var = m2_ave - m_ave * m_ave;
  double m2_var = m4_ave - m2_ave * m2_ave;
  double m4_var = m8_ave - m4_ave * m4_ave;

  double m_err = sqrt(m_var / double(n_update));
  double m2_err = sqrt(m2_var / double(n_update));
  double m4_err = sqrt(m4_var / double(n_update));

  double U4 = 1.5 * (1.0 - m4_ave / (3.0 * m2_ave * m2_ave));
  double U4_err = (m2_err * m4_ave / m2_ave + 0.5 * m4_err) / (m2_ave * m2_ave);

  printf("<cluster_size>/N=%.12e\n", cluster_ave);
  printf("<m>=%.12e (%.12e)\n", m_ave, m_err);
  printf("<m^2>=%.12e (%.12e)\n", m2_ave, m2_err);
  printf("<m^4>=%.12e (%.12e)\n", m4_ave, m4_err);
  printf("<U^4>=%.12e (%.12e)\n", U4, U4_err);

  return 0;
}
