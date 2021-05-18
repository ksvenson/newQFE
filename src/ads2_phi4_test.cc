// ads2_phi4_test.cc

#include <stdio.h>
#include <cmath>
#include <map>
#include "ads2.h"
#include "phi4.h"
#include "statistics.h"

int main(int argc, char* argv[]) {

  int N = 8;
  printf("N: %d\n", N);

  int q = 7;
  printf("q: %d\n", q);

  double musq = 0.0;
  printf("musq: %.4f\n", musq);

  double lambda = 0.0;
  printf("lambda: %.4f\n", lambda);

  QfeLatticeAdS2 lattice(N, q);
  printf("total sites: %ld\n", lattice.sites.size());
  printf("bulk sites: %d\n", lattice.n_bulk);
  printf("boundary sites: %d\n", lattice.n_boundary);
  printf("fixed sites: %d\n", lattice.n_fixed);

  QfePhi4 field(&lattice, musq, lambda);
  field.HotStart();

  printf("Initial Action: %.12f\n", field.Action());

  int n_therm = 100;
  int n_traj = 10000;
  int n_skip = 10;
  int n_wolff = 0;
  int n_metropolis = 10;

  // measurements
  std::vector<double> mag;
  std::vector<double> action;
  std::vector<double> cluster_size;
  std::vector<double> accept_metropolis;

  // boundary-boundary propagator (all to all)
  int n_bb = (lattice.n_bulk * (lattice.n_bulk + 1)) / 2;
  std::vector<double> bins_bb(n_bb);  // bin for each pair of sites
  std::vector<int> s1_bb(n_bb);  // s1 for each pair of sites
  std::vector<int> s2_bb(n_bb);  // s2 for each pair of sites
  std::map<int,int> sigma_map;  // map from a sigma value to its bin
  std::vector<double> sigma_bb;  // sigma for each bin

  int s1 = 0;
  int s2 = 0;
  for (int i = 0; i < n_bb; i++) {

    // bin in log(sigma), i.e. pairs of bulk points with log(sigma) within
    // 1e-3 of each other are in the same bin
    double sigma = lattice.Sigma(s1, s2);
    int log_sigma_int = int(log(sigma) / 1.0e-3);
    if (sigma_map.find(log_sigma_int) == sigma_map.end()) {
      // new bin
      bins_bb[i] = sigma_bb.size();
      sigma_map[log_sigma_int] = sigma_bb.size();
      sigma_bb.push_back(sigma);
    } else {
      // bin already exists
      bins_bb[i] = sigma_map[log_sigma_int];
    }

    s1_bb[i] = s1;
    s2_bb[i] = s2;
    s2++;
    if (s2 == lattice.n_bulk) {
      s2 = 0;
      s1++;
    }
  }
  std::vector<double> propagator_bb(sigma_bb.size(), 0.0);
  std::vector<int> n_meas_bb(sigma_bb.size());

  for (int n = 0; n < (n_traj + n_therm); n++) {

    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      cluster_size_sum += field.WolffUpdate();
    }
    double metropolis_sum = 0.0;
    for (int j = 0; j < n_metropolis; j++) {
      metropolis_sum += field.Metropolis();
    }
    cluster_size.push_back(double(cluster_size_sum) / double(lattice.n_bulk + lattice.n_boundary));
    accept_metropolis.push_back(metropolis_sum);

    if (n % n_skip || n < n_therm) continue;

    action.push_back(field.Action());
    mag.push_back(field.MeanPhi());

    // measure propagators
    for (int i = 0; i < n_bb; i++) {
      int bin = bins_bb[i];
      propagator_bb[bin] += field.phi[s1_bb[i]] * field.phi[s2_bb[i]];
      n_meas_bb[bin]++;
    }

    printf("%06d %.12f %+.12f %.4f %d\n", \
        n, action.back(), mag.back(), \
        accept_metropolis.back(), \
        cluster_size_sum);
  }

  // print average propagator for each bin
  printf("\nsigma / propagator_bb\n");
  for (int i = 0; i < sigma_bb.size(); i++) {
    printf("{%.12e, %.12e},\n", sigma_bb[i], \
        propagator_bb[i] / double(n_meas_bb[i]));
  }

  // compute magnetic moments
  std::vector<double> mag_abs(mag.size());
  std::vector<double> mag2(mag.size());
  std::vector<double> mag4(mag.size());
  for (int i = 0; i < mag.size(); i++) {
    double m = mag[i];
    double m2 = m * m;
    mag_abs[i] = abs(m);
    mag2[i] = m2;
    mag4[i] = m2 * m2;
  }

  // print statistics
  printf("accept_metropolis: %.4f\n", Mean(accept_metropolis));
  printf("cluster_size/V: %.4f\n", Mean(cluster_size));
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
