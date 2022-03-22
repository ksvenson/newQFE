// ising_tri_crit.cc

#include <getopt.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "ising.h"
#include "statistics.h"
#include "timer.h"

int main(int argc, char* argv[]) {

  // lattice size
  int N = 32;

  // triangle side lengths
  double l1 = 1.0;
  double l2 = 1.0;
  double l3 = 1.0;

  unsigned int seed = 1234u;
  int n_therm = 2000;
  int n_traj = 50000;
  int n_skip = 20;
  int n_wolff = 3;
  int n_metropolis = 5;
  double wall_time = 0.0;
  std::string data_dir = "ising_tri_crit";

  const struct option long_options[] = {
    { "n_lattice", required_argument, 0, 'N' },
    { "l1", required_argument, 0, 'a' },
    { "l2", required_argument, 0, 'b' },
    { "l3", required_argument, 0, 'c' },
    { "seed", required_argument, 0, 'S' },
    { "n_therm", required_argument, 0, 'h' },
    { "n_traj", required_argument, 0, 't' },
    { "n_skip", required_argument, 0, 's' },
    { "n_wolff", required_argument, 0, 'w' },
    { "n_metropolis", required_argument, 0, 'e' },
    { "wall_time", required_argument, 0, 'W' },
    { "data_dir", required_argument, 0, 'd' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "N:a:b:c:S:h:t:s:w:e:W:d:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N': N = atoi(optarg); break;
      case 'a': l1 = std::stod(optarg); break;
      case 'b': l2 = std::stod(optarg); break;
      case 'c': l3 = std::stod(optarg); break;
      case 'S': seed = atol(optarg); break;
      case 'h': n_therm = atoi(optarg); break;
      case 't': n_traj = atoi(optarg); break;
      case 's': n_skip = atoi(optarg); break;
      case 'w': n_wolff = atoi(optarg); break;
      case 'e': n_metropolis = atoi(optarg); break;
      case 'W': wall_time = std::stod(optarg); break;
      case 'd': data_dir = optarg; break;
      default: break;
    }
  }

  printf("N: %d\n", N);
  printf("l1: %.12f\n", l1);
  printf("l2: %.12f\n", l2);
  printf("l3: %.12f\n", l3);
  printf("seed: 0x%08X\n", seed);
  printf("n_therm: %d\n", n_therm);
  printf("n_traj: %d\n", n_traj);
  printf("n_skip: %d\n", n_skip);
  printf("n_wolff: %d\n", n_wolff);
  printf("n_metropolis: %d\n", n_metropolis);
  printf("wall_time: %f\n", wall_time);
  printf("data_dir: %s\n", data_dir.c_str());

  // find the dual lattice lengths
  double tri_area = 0.25 * sqrt((l1 + l2 + l3) * (l1 + l2 - l3) \
      * (l1 - l2 + l3) * (-l1 + l2 + l3));
  double l_sq_sum = l1 * l1 + l2 * l2 + l3 * l3;
  double ls1 = 0.25 * (l_sq_sum - 2.0 * l1 * l1) / tri_area;
  double ls2 = 0.25 * (l_sq_sum - 2.0 * l2 * l2) / tri_area;
  double ls3 = 0.25 * (l_sq_sum - 2.0 * l3 * l3) / tri_area;

  // compute the critical couplings
  double K1 = 0.5 * asinh(ls1);
  double K2 = 0.5 * asinh(ls2);
  double K3 = 0.5 * asinh(ls3);

  printf("K1: %.12f\n", K1);
  printf("K2: %.12f\n", K2);
  printf("K3: %.12f\n", K3);

  QfeLattice lattice;
  lattice.SeedRng(seed);
  lattice.InitTriangle(N, K1, K2, K3);

  QfeIsing field(&lattice, 1.0);
  field.HotStart();

  double vol = double(lattice.n_sites);

  printf("initial action: %.12f\n", field.Action());

  // measurements
  QfeMeasReal mag;  // average spin (magnetization)
  QfeMeasReal mag_2;  // magnetization^2
  QfeMeasReal mag_4;  // magnetization^4
  QfeMeasReal mag_6;  // magnetization^6
  QfeMeasReal mag_8;  // magnetization^8
  QfeMeasReal energy;  // energy
  QfeMeasReal energy_2;  // energy^2
  QfeMeasReal energy_4;  // energy^4
  QfeMeasReal mag_energy;  // magnetization * energy
  QfeMeasReal mag_2_energy;  // magnetization^2 * energy
  QfeMeasReal mag_4_energy;  // magnetization^4 * energy
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;

  Timer timer;

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
    int n_clusters = field.SWUpdate();

    double s2 = 0.0;
    double s4 = 0.0;
    double s6 = 0.0;
    double s8 = 0.0;
    for (int c = 0; c < n_clusters; c++) {
      double x = double(field.sw_clusters[c].size()) / vol;
      double x2 = x * x;
      double x4 = x2 * x2;
      s2 += x2;
      s4 += x4;
      s6 += x2 * x4;
      s8 += x4 * x4;
    }
    double s2_2 = s2 * s2;

    double m2 = s2;
    double m4 = 3.0 * s2_2 - 2.0 * s4;
    double m6 = 15.0 * s2_2 * s2 - 30.0 * s2 * s4 + 16.0 * s6;
    double m8 = 105.0 * s2_2 * s2_2 - 420.0 * s2_2 * s4 \
        + 140.0 * s4 * s4 + 448.0 * s2 * s6 - 272.0 * s8;

    double e_sum = 0.0;
    for (int l = 0; l < lattice.n_links; l++) {
      int s_a = lattice.links[l].sites[0];
      int s_b = lattice.links[l].sites[1];
      e_sum -= field.spin[s_a] * field.spin[s_b] * lattice.links[l].wt;
    }
    double e_mean = e_sum / vol;

    double m_sum = 0.0;
    for (int s = 0; s < lattice.n_sites; s++) {
      m_sum += field.spin[s];
    }
    double m_mean = m_sum / vol;

    // double m_sq = m_mean * m_mean;
    double e_sq = e_mean * e_mean;
    mag.Measure(fabs(m_mean));
    mag_2.Measure(m2);
    mag_4.Measure(m4);
    mag_6.Measure(m6);
    mag_8.Measure(m8);
    energy.Measure(e_mean);
    energy_2.Measure(e_sq);
    energy_4.Measure(e_sq * e_sq);
    mag_energy.Measure(fabs(m_mean) * e_mean);
    mag_2_energy.Measure(m2 * e_mean);
    mag_4_energy.Measure(m4 * e_mean);

    // printf("%06d %.12f %+.12f %.4f %.4f\n", \
    //     n, energy.last, mag.last, \
    //     accept_metropolis.last, \
    //     cluster_size.last);

    if (wall_time > 0.0 && timer.Duration() > wall_time) break;
  }

  timer.Stop();
  printf("duration: %.6f\n", timer.Duration());

  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());

  double m_mean = mag.Mean();
  double m_err = mag.Error();
  double m2_mean = mag_2.Mean();
  double m2_err = mag_2.Error();
  double m4_mean = mag_4.Mean();
  double m4_err = mag_4.Error();

  double U4_mean = 1.5 * (1.0 - m4_mean / (3.0 * m2_mean * m2_mean));
  double U4_err = 0.5 * U4_mean * sqrt(pow(m4_err / m4_mean, 2.0) + pow(2.0 * m2_err / m2_mean, 2.0));
  printf("U4: %.12e %.12e\n", U4_mean, U4_err);

  double m_susc_mean = m2_mean - m_mean * m_mean;
  double m_susc_err = sqrt(pow(m2_err, 2.0) + pow(2.0 * m_mean * m_err, 2.0));
  printf("m_susc: %.12e %.12e\n", m_susc_mean, m_susc_err);

  // open an output file
  char run_id[50];
  char path[200];
  sprintf(run_id, "%d_%.3f_%.3f_%.3f", N, l1, l2, l3);
  sprintf(path, "%s/%s/%s_%08X.dat", data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", path);
  FILE* file = fopen(path, "w");
  assert(file != nullptr);

  printf("mag: %.12e %.12e %.4f %.4f\n", \
      m_mean, m_err, \
      mag.AutocorrFront(), mag.AutocorrBack());
  fprintf(file, "mag %.12e %.12e %d\n", \
      m_mean, m_err, mag.n);
  printf("m^2: %.12e %.12e %.4f %.4f\n", \
      m2_mean, m2_err, \
      mag_2.AutocorrFront(), mag_2.AutocorrBack());
  fprintf(file, "mag^2 %.12e %.12e %d\n", \
      m2_mean, m2_err, mag_2.n);
  printf("m^4: %.12e %.12e %.4f %.4f\n", \
      m4_mean, m4_err, \
      mag_4.AutocorrFront(), mag_4.AutocorrBack());
  fprintf(file, "mag^4 %.12e %.12e %d\n", \
      m4_mean, m4_err, mag_4.n);
  printf("m^6: %.12e %.12e %.4f %.4f\n", \
      mag_6.Mean(), mag_6.Error(), \
      mag_6.AutocorrFront(), mag_6.AutocorrBack());
  fprintf(file, "mag^6 %.12e %.12e %d\n", \
      mag_6.Mean(), mag_6.Error(), mag_6.n);
  printf("m^8: %.12e %.12e %.4f %.4f\n", \
      mag_8.Mean(), mag_8.Error(), \
      mag_8.AutocorrFront(), mag_8.AutocorrBack());
  fprintf(file, "mag^8 %.12e %.12e %d\n", \
      mag_8.Mean(), mag_8.Error(), mag_8.n);
  printf("energy: %+.12e %.12e %.4f %.4f\n", \
      energy.Mean(), energy.Error(), \
      energy.AutocorrFront(), energy.AutocorrBack());
  fprintf(file, "energy %.12e %.12e %d\n", \
      energy.Mean(), energy.Error(), energy.n);
  printf("energy_2: %+.12e %.12e %.4f %.4f\n", \
      energy_2.Mean(), energy_2.Error(), \
      energy_2.AutocorrFront(), energy_2.AutocorrBack());
  fprintf(file, "energy^2 %.12e %.12e %d\n", \
      energy_2.Mean(), energy_2.Error(), energy_2.n);
  printf("energy_4: %+.12e %.12e %.4f %.4f\n", \
      energy_4.Mean(), energy_4.Error(), \
      energy_4.AutocorrFront(), energy_4.AutocorrBack());
  fprintf(file, "energy^4 %.12e %.12e %d\n", \
      energy_4.Mean(), energy_4.Error(), energy_4.n);
  printf("mag_energy: %+.12e %.12e %.4f %.4f\n", \
      mag_energy.Mean(), mag_energy.Error(), \
      mag_energy.AutocorrFront(), mag_energy.AutocorrBack());
  fprintf(file, "mag*energy %.12e %.12e %d\n", \
      mag_energy.Mean(), mag_energy.Error(), mag_energy.n);
  printf("mag_2_energy: %+.12e %.12e %.4f %.4f\n", \
      mag_2_energy.Mean(), mag_2_energy.Error(), \
      mag_2_energy.AutocorrFront(), mag_2_energy.AutocorrBack());
  fprintf(file, "mag^2*energy %.12e %.12e %d\n", \
      mag_2_energy.Mean(), mag_2_energy.Error(), mag_2_energy.n);
  printf("mag_4_energy: %+.12e %.12e %.4f %.4f\n", \
      mag_4_energy.Mean(), mag_4_energy.Error(), \
      mag_4_energy.AutocorrFront(), mag_4_energy.AutocorrBack());
  fprintf(file, "mag^4*energy %.12e %.12e %d\n", \
      mag_4_energy.Mean(), mag_4_energy.Error(), mag_4_energy.n);

  fclose(file);

  return 0;
}
