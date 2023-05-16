// ising_tri_crit.cc

#include <getopt.h>

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ising.h"
#include "statistics.h"
#include "timer.h"

int main(int argc, char* argv[]) {
  // lattice size
  int N = 6;

  // coupling constants
  double K1 = 1.0;
  double K2 = 1.0;
  double K3 = 1.0;
  double K4 = 1.0;
  double beta = 1.0;

  unsigned int seed = 1234u;
  int n_therm = 2000;
  int n_traj = 50000;
  int n_skip = 20;
  int n_wolff = 3;
  int n_metropolis = 5;
  double wall_time = 0.0;
  int max_window = 8;
  std::string data_dir = "ising_tri_crit_3d";

  const struct option long_options[] = {
      {"n_lattice", required_argument, 0, 'N'},
      {"K1", required_argument, 0, 'a'},
      {"K2", required_argument, 0, 'b'},
      {"K3", required_argument, 0, 'c'},
      {"K4", required_argument, 0, 'd'},
      {"beta", required_argument, 0, 'B'},
      {"seed", required_argument, 0, 'S'},
      {"n_therm", required_argument, 0, 'h'},
      {"n_traj", required_argument, 0, 't'},
      {"n_skip", required_argument, 0, 's'},
      {"n_wolff", required_argument, 0, 'w'},
      {"n_metropolis", required_argument, 0, 'e'},
      {"wall_time", required_argument, 0, 'W'},
      {"data_dir", required_argument, 0, 'D'},
      {0, 0, 0, 0}};

  const char* short_options = "N:a:b:c:d:B:S:h:t:s:w:e:W:D:";

  while (true) {
    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N':
        N = atoi(optarg);
        break;
      case 'a':
        K1 = std::stod(optarg);
        break;
      case 'b':
        K2 = std::stod(optarg);
        break;
      case 'c':
        K3 = std::stod(optarg);
        break;
      case 'd':
        K4 = std::stod(optarg);
        break;
      case 'B':
        beta = std::stod(optarg);
        break;
      case 'S':
        seed = atol(optarg);
        break;
      case 'h':
        n_therm = atoi(optarg);
        break;
      case 't':
        n_traj = atoi(optarg);
        break;
      case 's':
        n_skip = atoi(optarg);
        break;
      case 'w':
        n_wolff = atoi(optarg);
        break;
      case 'e':
        n_metropolis = atoi(optarg);
        break;
      case 'W':
        wall_time = std::stod(optarg);
        break;
      case 'D':
        data_dir = optarg;
        break;
      default:
        break;
    }
  }

  printf("N: %d\n", N);
  printf("K1: %.12f\n", K1);
  printf("K2: %.12f\n", K2);
  printf("K3: %.12f\n", K3);
  printf("K4: %.12f\n", K4);
  printf("beta: %.12f\n", beta);
  printf("seed: 0x%08X\n", seed);
  printf("n_therm: %d\n", n_therm);
  printf("n_traj: %d\n", n_traj);
  printf("n_skip: %d\n", n_skip);
  printf("n_wolff: %d\n", n_wolff);
  printf("n_metropolis: %d\n", n_metropolis);
  printf("wall_time: %f\n", wall_time);
  printf("data_dir: %s\n", data_dir.c_str());

  QfeLattice lattice;
  lattice.SeedRng(seed);
  lattice.InitTriangle(N, K1, K2, K3);
  lattice.AddDimension(N);

  // set the weights for the links in the z direction
  for (int s = 0; s < lattice.n_sites; s++) {
    int sp1 = (s + N * N) % (N * N * N);
    int l = lattice.FindLink(s, sp1);
    lattice.links[l].wt = K4;
  }

  QfeIsing field(&lattice, beta);
  field.HotStart();

  double vol = double(lattice.n_sites);

  printf("initial action: %.12f\n", field.Action());

  // measurements
  QfeMeasReal mag;     // average spin (magnetization)
  QfeMeasReal mag_2;   // magnetization^2
  QfeMeasReal mag_4;   // magnetization^4
  QfeMeasReal mag_6;   // magnetization^6
  QfeMeasReal mag_8;   // magnetization^8
  QfeMeasReal mag_10;  // magnetization^10
  QfeMeasReal mag_12;  // magnetization^12
  QfeMeasReal action;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;
  int n_corr = (max_window * 2 + 1) * (max_window * 2 + 1) * (max_window + 1);
  std::vector<QfeMeasReal> corr(n_corr);

  Timer timer;

  for (int n = 0; n < (n_traj + n_therm); n++) {
    if (wall_time > 0.0 && timer.Duration() > wall_time) break;

    double metropolis_sum = 0.0;
    for (int j = 0; j < n_metropolis; j++) {
      metropolis_sum += field.Metropolis();
    }
    accept_metropolis.Measure(metropolis_sum);

    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      cluster_size_sum += field.WolffUpdate();
    }
    cluster_size.Measure(double(cluster_size_sum) / vol);

    if (n % n_skip || n < n_therm) continue;

    // measure correlator values
    std::vector<int> corr_sum(n_corr, 0);
    for (int i1 = 0; i1 < field.wolff_cluster.size(); i1++) {
      int s1 = field.wolff_cluster[i1];
      int x1 = s1 % N;
      int y1 = (s1 / N) % N;
      int z1 = s1 / (N * N);
      double s1_spin = field.spin[i1];
      for (int i2 = 0; i2 < field.wolff_cluster.size(); i2++) {
        int s2 = field.wolff_cluster[i2];
        int x2 = s2 % N;
        int y2 = (s2 / N) % N;
        int z2 = s2 / (N * N);
        double s2_spin = field.spin[i2];

        // dx goes left and right
        int dx = (x2 - x1 + N) % N;
        if (dx > N / 2) dx = dx - N;
        if (abs(dx) > max_window) continue;

        // dy goes up and down
        int dy = (y2 - y1 + N) % N;
        if (dy > N / 2) dy = dy - N;
        if (abs(dy) > max_window) continue;

        // dz only goes up
        int dz = (z2 - z1 + N) % N;
        if (dz > max_window) continue;

        int i_corr = dx + max_window +
                     (2 * max_window + 1) *
                         (dy + max_window + (2 * max_window + 1) * dz);
        if (s1_spin == s2_spin) {
          corr_sum[i_corr]++;
        } else {
          corr_sum[i_corr]--;
        }
      }
    }

    double corr_vol = double(field.wolff_cluster.size());
    for (int i = 0; i < n_corr; i++) {
      corr[i].Measure(double(corr_sum[i]) / corr_vol);
    }

    double spin_sum = 0.0;
    for (int s = 0; s < lattice.n_sites; s++) {
      spin_sum += field.spin[s];
    }
    double m = spin_sum / vol;
    double m_sq = m * m;
    mag.Measure(fabs(m));
    mag_2.Measure(m_sq);
    mag_4.Measure(m_sq * m_sq);
    mag_6.Measure(mag_4.last * m_sq);
    mag_8.Measure(mag_6.last * m_sq);
    mag_10.Measure(mag_8.last * m_sq);
    mag_12.Measure(mag_10.last * m_sq);
    action.Measure(field.Action());
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
  double U4_err =
      0.5 * U4_mean *
      sqrt(pow(m4_err / m4_mean, 2.0) + pow(2.0 * m2_err / m2_mean, 2.0));
  printf("U4: %.12e %.12e\n", U4_mean, U4_err);

  double m_susc_mean = m2_mean - m_mean * m_mean;
  double m_susc_err = sqrt(pow(m2_err, 2.0) + pow(2.0 * m_mean * m_err, 2.0));
  printf("m_susc: %.12e %.12e\n", m_susc_mean, m_susc_err);

  // open an output file
  char run_id[50];
  char path[200];
  sprintf(run_id, "%d_%.6f_%.3f_%.3f_%.3f_%.3f", N, beta, K1, K2, K3, K4);
  sprintf(path, "%s/%s_%08X.dat", data_dir.c_str(), run_id, seed);
  printf("opening file: %s\n", path);
  FILE* file = fopen(path, "w");
  assert(file != nullptr);

  for (int i = 0; i < n_corr; i++) {
    int dx = i % (max_window * 2 + 1);
    int dy = (i / (max_window * 2 + 1)) % (max_window * 2 + 1);
    int dz = i / ((max_window * 2 + 1) * (max_window * 2 + 1));
    fprintf(file, "%+03d %+03d %02d ", dx - max_window, dy - max_window, dz);
    corr[i].WriteMeasurement(file);
  }

  fclose(file);

  return 0;
}
