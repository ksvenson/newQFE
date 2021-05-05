// lattice_test.cc

#include <stdio.h>
#include "lattice.h"
#include "wolff.h"
#include "overrelax.h"
#include "metropolis.h"

double Mean(std::vector<double>& a);
double U4(std::vector<double>& m2, std::vector<double>& m4);
double JackknifeMean(std::vector<double>& a);
double JackknifeU4(std::vector<double>& m2, std::vector<double>& m4);
double AutocorrGamma(std::vector<double>& a, int n);
double AutocorrTime(std::vector<double>& a);

int main(int argc, char* argv[]) {

  int N = 64;
  printf("N: %d\n", N);

  double skew = 1.0;
  printf("skew: %.2f\n", skew);

  QfeLattice lattice;
  lattice.InitTriangle(N, skew);
  lattice.HotStart();

  QfeMetropolis metropolis;
  metropolis.Init(&lattice);

  QfeOverrelax overrelax;
  overrelax.Init(&lattice);

  QfeWolff wolff;
  wolff.Init(&lattice);

  printf("Initial Action: %.12f\n", lattice.Action());

  // measurements
  std::vector<double> mag;
  std::vector<double> action;
  std::vector<double> demon;
  std::vector<double> cluster_size;
  std::vector<double> accept_metropolis;
  std::vector<double> accept_overrelax;

  int n_traj = 20000;
  int n_therm = 1000;
  int n_skip = 10;
  int n_wolff = 4;
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

  std::vector<double> mag2(mag.size());
  std::vector<double> mag4(mag.size());
  for (int i = 0; i < mag.size(); i++) {
    double m = mag[i];
    double m2 = m * m;
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

  return 0;
}

double Mean(std::vector<double>& a) {
  double sum = 0.0;
  for (int i = 0; i < a.size(); i++) sum += a[i];
  return sum / double(a.size());
}

double U4(std::vector<double>& m2, std::vector<double>& m4) {
  double m2_mean = Mean(m2);
  double m4_mean = Mean(m4);

  return 1.5 * (1.0 - m4_mean / (3.0 * m2_mean * m2_mean));
}

double JackknifeMean(std::vector<double>& a) {
  int n = a.size();
	double mean = Mean(a);
	double err = 0.0;

	for (int i = 0; i < n; i++) {
    std::vector<double> a_del = a;
    a_del.erase(a_del.begin() + i);
    double diff = Mean(a_del) - mean;
    err += diff * diff;
  }

	err = sqrt((double(n) - 1.0) / double(n) * err);
	return err;
}

double JackknifeU4(std::vector<double>& m2, std::vector<double>& m4) {
  int n = m2.size();
	double mean = U4(m2, m4);
	double err = 0.0;

	for (int i = 0; i < n; i++) {
    std::vector<double> m2_del = m2;
    std::vector<double> m4_del = m4;
    m2_del.erase(m2_del.begin() + i);
    m4_del.erase(m4_del.begin() + i);
    double diff = U4(m2_del, m4_del) - mean;
    err += diff * diff;
  }

	err = sqrt((double(n) - 1.0) / double(n) * err);
	return err;
}

double AutocorrGamma(std::vector<double>& a, int n) {
  int N = a.size();
  double result = 0.0;
  double mean = Mean(a);
  int start = 0;
  int end = N - n;

  if (n < 0) {
    start = -n;
    end = N;
  }

  for (int i = start; i < end; i++) {
    result += (a[i] - mean) * (a[i + n] - mean);
  }

  return result / double(end - start);
}

double AutocorrTime(std::vector<double>& a) {
  double Gamma0 = AutocorrGamma(a, 0);
  double result = 0.5 * Gamma0;

  for (int n = 1; n < a.size(); n++) {
    double curGamma = AutocorrGamma(a, n);
    if (curGamma < 0.0) break;
    result += curGamma;
  }

  return result / Gamma0;
}
