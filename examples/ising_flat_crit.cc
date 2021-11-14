// ising_flat_crit.cc

#include <getopt.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "ising.h"
#include "statistics.h"

double tri_crit(double k1, double k2, double k3, double beta) {
  double p1 = exp(-2.0 * beta * (k2 + k3));
  double p2 = exp(-2.0 * beta * (k3 + k1));
  double p3 = exp(-2.0 * beta * (k1 + k2));

  // calculate the residual and its derivative wrt beta
  double r1 = p1 + p2 + p3 - 1.0;
  double r2 = -2.0 * (p1 * (k2 + k3) + p2 * (k3 + k1) + p3 * (k1 + k2));
  return r1 / r2;
}

double find_crit(double k1, double k2, double k3) {

  // normalize the couplings
  double k_mean = (k1 + k2 + k3) / 3.0;
  k1 /= k_mean;
  k2 /= k_mean;
  k3 /= k_mean;

  // start with the equilateral critical value
  double beta = 0.267949192431123;  // 2 - sqrt(3)

  // do 100 iterations of newton's method
  // it normally converges in less that 10 iterations
  for (int i = 0; i < 100; i++) {
    beta -= tri_crit(k1, k2, k3, beta);
    // printf("%04d %.20f\n", i, beta);
  }

  // return the unnormalized critical value of beta
  return beta / k_mean;
}

int main(int argc, char* argv[]) {

  int N = 32;

  // choose weights for the 3 directions and calculate beta critical
  double k1 = 1.0;
  double k2 = 1.0;
  double k3 = 1.0;

  // multiplier to move above or below the critical point
  double beta_mult = 1.0;

  int n_therm = 2000;
  int n_traj = 50000;
  int n_skip = 20;
  int n_wolff = 3;
  int n_metropolis = 5;

  const struct option long_options[] = {
    { "n_lattice", required_argument, 0, 'N' },
    { "k1", required_argument, 0, 'a' },
    { "k2", required_argument, 0, 'b' },
    { "k3", required_argument, 0, 'c' },
    { "beta_mult", required_argument, 0, 'm' },
    { "n_therm", required_argument, 0, 'h' },
    { "n_traj", required_argument, 0, 't' },
    { "n_skip", required_argument, 0, 's' },
    { "n_wolff", required_argument, 0, 'w' },
    { "n_metropolis", required_argument, 0, 'e' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "N:a:b:c:m:h:t:s:w:e:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N': N = atoi(optarg); break;
      case 'a': k1 = std::stod(optarg); break;
      case 'b': k2 = std::stod(optarg); break;
      case 'c': k3 = std::stod(optarg); break;
      case 'm': beta_mult = std::stod(optarg); break;
      case 'h': n_therm = atoi(optarg); break;
      case 't': n_traj = atoi(optarg); break;
      case 's': n_skip = atoi(optarg); break;
      case 'w': n_wolff = atoi(optarg); break;
      case 'e': n_metropolis = atoi(optarg); break;
      default: break;
    }
  }

  printf("N: %d\n", N);
  printf("k1: %.12f\n", k1);
  printf("k2: %.12f\n", k2);
  printf("k3: %.12f\n", k3);
  printf("beta_mult: %.12f\n", beta_mult);

  QfeLattice lattice;
  lattice.InitTriangle(N, k1, k2, k3);

  QfeIsing field(&lattice, find_crit(k1, k2, k3) * beta_mult);
  field.HotStart();

  printf("beta: %.12f\n", field.beta);
  printf("initial action: %.12f\n", field.Action());

  // correlator measurements in each direction
  std::vector<QfeMeasReal> corr_x(N/2);
  std::vector<QfeMeasReal> corr_y(N/2);
  std::vector<QfeMeasReal> corr_z(N/2);
  std::vector<QfeMeasReal> corr_w(N/2);
  std::vector<QfeMeasReal> corr_xz(N/2);
  std::vector<QfeMeasReal> corr_yz(N/2);

  // correlator averaged over zero momentum mode
  std::vector<QfeMeasReal> corr_zero_x(N/2);
  std::vector<QfeMeasReal> corr_zero_y(N/2);
  std::vector<QfeMeasReal> corr_zero_z(N/2);

  // measurements
  std::vector<double> mag;
  std::vector<double> action;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;

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

    // measure correlators
    std::vector<int> corr_x_sum(N, 0);
    std::vector<int> corr_y_sum(N, 0);
    std::vector<int> corr_z_sum(N, 0);
    std::vector<int> corr_w_sum(N, 0);
    std::vector<int> corr_xz_sum(N, 0);
    std::vector<int> corr_yz_sum(N, 0);
    std::vector<int> corr_zero_x_sum(N, 0);
    std::vector<int> corr_zero_y_sum(N, 0);
    std::vector<int> corr_zero_z_sum(N, 0);
    int count = field.wolff_cluster.size();
    for (int i1 = 0; i1 < count; i1++) {
      for (int i2 = i1; i2 < count; i2++) {

        int s1 = field.wolff_cluster[i1];
        int x1 = s1 % N;
        int y1 = s1 / N;
        int z1 = (x1 + y1) % N;
        int w1 = (x1 - y1 + N) % N;
        int xz1 = (x1 - 2 * y1 + 2 * N) % N;
        int yz1 = (y1 - 2 * x1 + 2 * N) % N;

        int s2 = field.wolff_cluster[i2];
        int x2 = s2 % N;
        int y2 = s2 / N;
        int z2 = (x2 + y2) % N;
        int w2 = (x2 - y2 + N) % N;
        int xz2 = (x2 - 2 * y2 + 2 * N) % N;
        int yz2 = (y2 - 2 * x2 + 2 * N) % N;

        int dx = (N - abs(2 * abs(x1 - x2) - N)) / 2;
        int dy = (N - abs(2 * abs(y1 - y2) - N)) / 2;
        int dw = (N - abs(2 * abs(w1 - w2) - N)) / 2;

        if (y1 == y2) corr_x_sum[dx]++;
        if (x1 == x2) corr_y_sum[dy]++;
        if (w1 == w2) corr_z_sum[dx]++;
        if (z1 == z2) corr_w_sum[dx]++;
        if (xz1 == xz2) corr_xz_sum[dy]++;
        if (yz1 == yz2) corr_yz_sum[dx]++;
        corr_zero_x_sum[dy]++;
        corr_zero_y_sum[dx]++;
        corr_zero_z_sum[dw]++;
      }
    }

    // add correlator measurements
    for (int i = 0; i < N/2; i++) {
      corr_x[i].Measure(double(corr_x_sum[i]) / double(count));
      corr_y[i].Measure(double(corr_y_sum[i]) / double(count));
      corr_z[i].Measure(double(corr_z_sum[i]) / double(count));
      corr_w[i].Measure(double(corr_w_sum[i]) / double(count));
      corr_xz[i].Measure(double(corr_xz_sum[i]) / double(count));
      corr_yz[i].Measure(double(corr_yz_sum[i]) / double(count));
      corr_zero_x[i].Measure(double(corr_zero_x_sum[i]) / double(count * N));
      corr_zero_y[i].Measure(double(corr_zero_y_sum[i]) / double(count * N));
      corr_zero_z[i].Measure(double(corr_zero_z_sum[i]) / double(count * N));
    }

    action.push_back(field.Action());
    mag.push_back(field.MeanSpin());
    printf("%06d %.12f %+.12f %.4f %.4f\n", \
        n, action.back(), mag.back(), \
        accept_metropolis.last, \
        cluster_size.last);
  }

  std::vector<double> mag_abs(mag.size());
  std::vector<double> mag2(mag.size());
  std::vector<double> mag4(mag.size());
  for (int i = 0; i < mag.size(); i++) {
    double m = mag[i];
    double m2 = m * m;
    mag_abs[i] = fabs(m);
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

  // open an output file
  char path[50];
  sprintf(path, "ising_flat_crit/%d_%.3f_%.3f_%.3f_.dat", N, k1, k2, k3);
  FILE* file = fopen(path, "w");
  assert(file != nullptr);

  printf("\ncorr_x:\n");
  for (int i = 0; i < N/2; i++) {
    printf("0 %04d %.12e %.12e\n", i, corr_x[i].Mean(), corr_x[i].Error());
    fprintf(file, "0 %04d %.12e %.12e\n", i, corr_x[i].Mean(), corr_x[i].Error());
  }

  printf("\ncorr_y:\n");
  for (int i = 0; i < N/2; i++) {
    printf("1 %04d %.12e %.12e\n", i, corr_y[i].Mean(), corr_y[i].Error());
    fprintf(file, "1 %04d %.12e %.12e\n", i, corr_y[i].Mean(), corr_y[i].Error());
  }

  printf("\ncorr_z:\n");
  for (int i = 0; i < N/2; i++) {
    printf("2 %04d %.12e %.12e\n", i, corr_z[i].Mean(), corr_z[i].Error());
    fprintf(file, "2 %04d %.12e %.12e\n", i, corr_z[i].Mean(), corr_z[i].Error());
  }

  printf("\ncorr_w:\n");
  for (int i = 0; i < N/2; i++) {
    printf("3 %04d %.12e %.12e\n", i, corr_w[i].Mean(), corr_w[i].Error());
    fprintf(file, "3 %04d %.12e %.12e\n", i, corr_w[i].Mean(), corr_w[i].Error());
  }

  printf("\ncorr_xz:\n");
  for (int i = 0; i < N/2; i++) {
    printf("4 %04d %.12e %.12e\n", i, corr_xz[i].Mean(), corr_xz[i].Error());
    fprintf(file, "4 %04d %.12e %.12e\n", i, corr_xz[i].Mean(), corr_xz[i].Error());
  }

  printf("\ncorr_yz:\n");
  for (int i = 0; i < N/2; i++) {
    printf("5 %04d %.12e %.12e\n", i, corr_yz[i].Mean(), corr_yz[i].Error());
    fprintf(file, "5 %04d %.12e %.12e\n", i, corr_yz[i].Mean(), corr_yz[i].Error());
  }

  printf("\ncorr_zero_x:\n");
  for (int i = 0; i < N/2; i++) {
    printf("6 %04d %.12e %.12e\n", i, corr_zero_x[i].Mean(), corr_zero_x[i].Error());
    fprintf(file, "6 %04d %.12e %.12e\n", i, corr_zero_x[i].Mean(), corr_zero_x[i].Error());
  }

  printf("\ncorr_zero_y:\n");
  for (int i = 0; i < N/2; i++) {
    printf("7 %04d %.12e %.12e\n", i, corr_zero_y[i].Mean(), corr_zero_y[i].Error());
    fprintf(file, "7 %04d %.12e %.12e\n", i, corr_zero_y[i].Mean(), corr_zero_y[i].Error());
  }

  printf("\ncorr_zero_z:\n");
  for (int i = 0; i < N/2; i++) {
    printf("8 %04d %.12e %.12e\n", i, corr_zero_z[i].Mean(), corr_zero_z[i].Error());
    fprintf(file, "8 %04d %.12e %.12e\n", i, corr_zero_z[i].Mean(), corr_zero_z[i].Error());
  }

  return 0;
}
