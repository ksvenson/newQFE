// s3_test.cc

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "s3.h"
#include "ising.h"
#include "statistics.h"

int main(int argc, char* argv[]) {

  int q = 5;
  printf("q: %d\n", q);

  double beta = 0.1;
  printf("beta: %.4f\n", beta);

  QfeLatticeS3 lattice(0);
  char lattice_path[50];
  sprintf(lattice_path, "s3_%d.dat", q);
  FILE* file = fopen(lattice_path, "r");
  assert(file != nullptr);
  lattice.ReadLattice(file);
  fclose(file);
  lattice.vol = double(lattice.n_sites);

  // for (int c = 0; c < lattice.n_cells; c++) {
  //   Eigen::Vector4d cr = lattice.CellCircumcenter(c);
  //   Eigen::Vector4d r_a = lattice.r[lattice.cells[c].sites[0]];
  //   printf("%04d %.16f\n", c, (cr - r_a).norm());
  // }
  //
  // for (int f = 0; f < lattice.n_faces; f++) {
  //   Eigen::Vector4d cr = lattice.FaceCircumcenter(f);
  //   Eigen::Vector4d r_a = lattice.r[lattice.faces[f].sites[0]];
  //   printf("%04d %.16f\n", f, (cr - r_a).norm());
  // }
  //
  // for (int l = 0; l < lattice.n_links; l++) {
  //   Eigen::Vector4d cr = lattice.EdgeCenter(l);
  //   Eigen::Vector4d r_a = lattice.r[lattice.links[l].sites[0]];
  //   printf("%04d %.16f\n", l, (cr - r_a).norm());
  // }

  // lattice.Inflate();
  // lattice.UpdateAntipodes();
  // exit(0);

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
    cluster_size.Measure(double(cluster_size_sum) / lattice.vol);
    accept_metropolis.Measure(metropolis_sum);

    if (n % n_skip || n < n_therm) continue;

    action.Measure(field.Action());
    double m = fabs(field.MeanSpin());
    double m2 = m * m;
    mag.Measure(m);
    mag_2.Measure(m2);
    mag_4.Measure(m2 * m2);
    printf("%06d %.12f %+.12f %.4f %.4f\n", \
        n, action.last, mag.last, \
        accept_metropolis.last, \
        cluster_size.last);
  }

  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());
  printf("cluster_size/V: %.4f\n", cluster_size.Mean());

  double m_mean = mag.Mean();
  double m_err = mag.Error();
  double m2_mean = mag_2.Mean();
  double m2_err = mag_2.Error();
  double m4_mean = mag_4.Mean();
  double m4_err = mag_4.Error();

  printf("action: %+.12e %.12e %.4f %.4f\n", \
      action.Mean(), action.Error(), \
      action.AutocorrFront(), action.AutocorrBack());
  printf("mag: %.12e %.12e %.4f %.4f\n", \
      m_mean, m_err, mag.AutocorrFront(), mag.AutocorrBack());
  printf("m^2: %.12e %.12e %.4f %.4f\n", \
      m2_mean, m2_err, mag_2.AutocorrFront(), mag_2.AutocorrBack());
  printf("m^4: %.12e %.12e %.4f %.4f\n", \
      m4_mean, m4_err, \
      mag_4.AutocorrFront(), mag_4.AutocorrBack());

  double U4_mean = 1.5 * (1.0 - m4_mean / (3.0 * m2_mean * m2_mean));
  double U4_err = 0.5 * U4_mean * sqrt(pow(m4_err / m4_mean, 2.0) \
      + pow(2.0 * m2_err / m2_mean, 2.0));
  printf("U4: %.12e %.12e\n", U4_mean, U4_err);

  double m_susc_mean = (m2_mean - m_mean * m_mean) * lattice.vol;
  double m_susc_err = sqrt(pow(m2_err, 2.0) \
      + pow(2.0 * m_mean * m_err, 2.0)) * lattice.vol;
  printf("m_susc: %.12e %.12e\n", m_susc_mean, m_susc_err);

  return 0;
}
