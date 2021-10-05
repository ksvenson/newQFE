// ising_flat_crit.cc

#include <cmath>
#include <cstdio>
#include <vector>
#include "ising.h"
#include "statistics.h"

double tri_crit(double k1, double k2, double k3, double beta) {
  double v1 = tanh(beta * k1);
  double v2 = tanh(beta * k2);
  double v3 = tanh(beta * k3);
  double n = (1.0 - v1 * v1) * (1.0 - v2 * v2) * (1.0 - v3 * v3);
  double d = (1.0 + v1 * v2 * v3) * (v1 + v2 * v3) * (v2 + v3 * v1) * (v3 + v1 * v2);

  double v12 = v1 * v2;
  double v13 = v1 * v3;
  double v23 = v2 * v3;
  double v1p = (v1 * v1 + 1.0);
  double v1m = (v1 * v1 - 1.0);
  double v2p = (v2 * v2 + 1.0);
  double v2m = (v2 * v2 - 1.0);
  double v3p = (v3 * v3 + 1.0);
  double v3m = (v3 * v3 - 1.0);

  double c1 = 2.0 * (v23 + v12 * v13) + v1 * v2p * v3p;
  double c2 = 2.0 * (v13 + v12 * v23) + v1p * v2 * v3p;
  double c3 = 2.0 * (v12 + v13 * v23) + v1p * v2p * v3;

  double s1 = 1.0 / cosh(beta * k1);
  double A1 = -s1 * s1 * v2m * v3m * c2 * c3;

  double s2 = 1.0 / cosh(beta * k2);
  double A2 = -s2 * s2 * v1m * v3m * c1 * c3;

  double s3 = 1.0 / cosh(beta * k3);
  double A3 = -s3 * s3 * v1m * v2m * c1 * c2;

  double crit = 0.25 * n / sqrt(d) - 1.0;
  double d_crit = 0.125 * (A1 + A2 + A3) / pow(d, 1.5);

  return crit / d_crit;
}

double find_crit(double k1, double k2, double k3) {
  double beta = 0.4;

  for (int i = 0; i < 100; i++) {
    beta = beta - tri_crit(k1, k2, k3, beta);
  }
  return beta;
}

int main(int argc, char* argv[]) {

  int N = 128;
  printf("N: %d\n", N);

  // choose weights for the 3 directions and calculate the critical
  // value of beta
  double k1 = 0.5;
  double k2 = 1.0;
  double k3 = 2.0;
  double beta = find_crit(k1, k2, k3);
  printf("beta: %.12f\n", beta);

  QfeLattice lattice;
  lattice.InitTriangle(N, k1, k2, k3);

  QfeIsing field(&lattice, beta);
  field.HotStart();

  printf("initial action: %.12f\n", field.Action());

  // correlator measurements in each direction
  std::vector<QfeMeasReal> corr_x(N/2);
  std::vector<QfeMeasReal> corr_y(N/2);
  std::vector<QfeMeasReal> corr_z(N/2);
  std::vector<QfeMeasReal> corr_w(N/2);
  std::vector<QfeMeasReal> corr_xz(N/2);
  std::vector<QfeMeasReal> corr_yz(N/2);

  // measurements
  std::vector<double> mag;
  std::vector<double> action;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;

  int n_therm = 1000;
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

        if (y1 == y2) corr_x_sum[dx]++;
        if (x1 == x2) corr_y_sum[dy]++;
        if (w1 == w2) corr_z_sum[dx]++;
        if (z1 == z2) corr_w_sum[dx]++;
        if (xz1 == xz2) corr_xz_sum[dy]++;
        if (yz1 == yz2) corr_yz_sum[dx]++;
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

  return 0;
}
