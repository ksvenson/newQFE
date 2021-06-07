// s2_test.cc

#include <cstdio>
#include <vector>
#include "phi4.h"
#include "s2.h"
#include "statistics.h"

int main(int argc, char* argv[]) {

  double musq = 1.8163 * 2;
  printf("musq: %.4f\n", musq);

  double lambda = 1.0;
  printf("lambda: %.4f\n", lambda);

  QfeLatticeS2 lattice;
  lattice.Refine2D(16);
  lattice.Inflate();
  lattice.UpdateWeights();
  lattice.CheckConnectivity();
  lattice.CheckConsistency();

  double site_wt_sum = 0.0;
  double site_wt_sq = 0.0;
  for (int s = 0; s < lattice.n_sites; s++) {
    site_wt_sum += lattice.sites[s].wt;
    site_wt_sq += lattice.sites[s].wt * lattice.sites[s].wt;
  }
  printf("site_wt_sum: %.12f\n", site_wt_sum);
  printf("site_wt_sq: %.12f\n", site_wt_sq);

  double link_wt_sum = 0.0;
  double link_wt_sq = 0.0;
  for (int l = 0; l < lattice.n_links; l++) {
    link_wt_sum += lattice.links[l].wt;
    link_wt_sq += lattice.links[l].wt * lattice.links[l].wt;
  }
  printf("link_wt_sum: %.12f\n", link_wt_sum);
  printf("link_wt_sq: %.12f\n", link_wt_sq);

  QfeMeasReal face_area;
  for (int f = 0; f < lattice.n_faces; f++) {
    face_area.Measure(lattice.FlatArea(f));
  }
  printf("face_area: %.12f (%.12f)\n", face_area.Mean(), face_area.Error());

  QfePhi4 field(&lattice, musq, lambda);
  field.HotStart();

  printf("initial action: %.12f\n", field.Action());

  // measurements
  std::vector<double> mag;
  std::vector<double> action;
  QfeMeasReal demon;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;
  QfeMeasReal accept_overrelax;

  int n_therm = 1000;
  int n_traj = 20000;
  int n_skip = 20;
  int n_wolff = 4;
  for (int n = 0; n < (n_traj + n_therm); n++) {

    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      cluster_size_sum += field.WolffUpdate();
    }
    cluster_size.Measure(double(cluster_size_sum) / double(lattice.n_sites));
    accept_metropolis.Measure(field.Metropolis());
    accept_overrelax.Measure(field.Overrelax());
    demon.Measure(field.overrelax_demon);

    if (n % n_skip || n < n_therm) continue;

    action.push_back(field.Action());
    mag.push_back(field.MeanPhi());
    printf("%06d %.12f %+.12f %.4f %.4f %.12f %.4f\n", \
        n, action.back(), mag.back(), \
        accept_metropolis.last, \
        accept_overrelax.last, demon.last, \
        cluster_size.last);
  }

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

  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());
  printf("accept_overrelax: %.4f\n", accept_overrelax.Mean());
  printf("demon: %.12f (%.12f)\n", demon.Mean(), demon.Error());
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

  return 0;
}
