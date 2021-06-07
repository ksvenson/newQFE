// ads2_phi4_test.cc

#include <cmath>
#include <cstdio>
#include <map>
#include <vector>
#include "ads2.h"
#include "phi4.h"
#include "statistics.h"

int main(int argc, char* argv[]) {

  int N = 6;
  printf("N: %d\n", N);

  int q = 7;
  printf("q: %d\n", q);

  double musq = 0.0;
  printf("musq: %.4f\n", musq);

  double lambda = 0.0;
  printf("lambda: %.4f\n", lambda);

  QfeLatticeAdS2 lattice(N, q);
  printf("total sites: %d\n", lattice.n_sites + lattice.n_dummy);
  printf("bulk sites: %d\n", lattice.n_bulk);
  printf("boundary sites: %d\n", lattice.n_boundary);
  printf("dummy sites: %d\n", lattice.n_dummy);

  QfePhi4 field(&lattice, musq, lambda);
  field.HotStart();

  printf("initial action: %.12f\n", field.Action());

  int n_therm = 1000;
  int n_traj = 10000;
  int n_skip = 10;
  int n_metropolis = 10;

  // measurements
  std::vector<double> mag;
  std::vector<double> action;
  QfeMeasReal accept_metropolis;

  // bulk-bulk 2-pt function (all to all)
  std::vector<int> bins_bb;  // bin for each pair of sites
  std::vector<int> s1_bb;  // s1 for each pair of sites
  std::vector<int> s2_bb;  // s2 for each pair of sites
  std::map<int,int> sigma_map;  // map from a sigma value to its bin
  std::vector<double> sigma_bb;  // sigma for each bin

  // only include pairs of points that are less than a certain distance apart
  double sigma_max = double(N);

  for (int i1 = 0, i2 = 1; i1 < (lattice.n_bulk - 2); i2++) {

    if (i2 == lattice.n_bulk) {
      i1++;
      i2 = i1 + 1;
    }
    int s1 = lattice.bulk_sites[i1];
    int s2 = lattice.bulk_sites[i2];

    // bin in sigma, i.e. pairs of bulk points with sigma within
    // 1e-6 of each other are in the same bin
    double sigma = lattice.Sigma(s1, s2);
    // printf("%04d %04d %.12f\n", s1, s2, sigma);
    if (sigma > sigma_max) continue;

    // don't include pairs of sites in different time slices (AdS3)
    // if (lattice.t[s1] != lattice.t[s2]) continue;

    // only include sites with the same rho as the first boundary site
    // if (abs(lattice.z[s1]) - abs(lattice.z[lattice.boundary_sites[0]]) > 1e-6) continue;

    // only include sites with the same spatial position (AdS3)
    // if (abs(lattice.z[s1] - lattice.z[s2]) > 1e-6) continue;

    // get an integer representation of sigma to use as a bin key
    int sigma_int = int(round(sigma / 1.0e-6));

    // check if the bin already exists
    if (sigma_map.find(sigma_int) == sigma_map.end()) {
      // create a new bin
      bins_bb.push_back(sigma_bb.size());
      sigma_map[sigma_int] = sigma_bb.size();
      sigma_bb.push_back(sigma);
    } else {
      // bin already exists
      bins_bb.push_back(sigma_map[sigma_int]);
    }

    s1_bb.push_back(s1);
    s2_bb.push_back(s2);
  }
  int n_bb = bins_bb.size();  // number of bulk-bulk pairs of sites
  printf("n_bb: %d\n", n_bb);
  std::vector<double> two_point_bb(sigma_bb.size(), 0.0);
  std::vector<double> two_point_bb_2(sigma_bb.size(), 0.0);
  std::vector<int> n_meas_bb(sigma_bb.size(), 0);

  for (int n = 0; n < (n_traj + n_therm); n++) {

    double metropolis_sum = 0.0;
    for (int j = 0; j < n_metropolis; j++) {
      metropolis_sum += field.Metropolis();
    }
    accept_metropolis.Measure(metropolis_sum);

    if (n % n_skip || n < n_therm) continue;

    action.push_back(field.Action());
    mag.push_back(field.MeanPhi());

    // measure bulk-bulk 2-pt functions
    for (int i = 0; i < n_bb; i++) {
      int bin = bins_bb[i];
      double bb = field.phi[s1_bb[i]] * field.phi[s2_bb[i]];
      two_point_bb[bin] += bb;
      two_point_bb_2[bin] += bb * bb;
      n_meas_bb[bin]++;
    }

    printf("%06d %.12f %+.12f %.4f\n", \
        n, action.back(), mag.back(), \
        accept_metropolis.last);
  }

  // print average 2-pt function for each bin
  printf("\nsigma / two_point_bb\n");
  for (int i = 0; i < sigma_bb.size(); i++) {
    double bb = two_point_bb[i] / double(n_meas_bb[i]);
    double bb_2 = two_point_bb_2[i] / double(n_meas_bb[i]);
    double bb_var = bb_2 - bb * bb;
    double bb_err = sqrt(bb_var / double(n_meas_bb[i]));
    printf("{%.12e, %.12e, %.12e},\n", sigma_bb[i], bb, bb_err);
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
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());
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
