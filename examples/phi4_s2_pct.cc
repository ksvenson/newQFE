// phi4_s2_pct.cc

#include <complex>
#include <cstdio>
#include <string>
#include <vector>
#include "phi4.h"
#include "s2.h"
#include "statistics.h"

typedef std::complex<double> Complex;

int main(int argc, char* argv[]) {

  int q = 3;
  int n_refine = 64;
  if (argc > 1) {
    n_refine = atoi(argv[1]);
  }
  printf("n_refine: %d\n", n_refine);

  double msq = -21.96; // -0.56; // -38.8; // -3.6326;
  if (argc > 2) {
    msq = std::stod(argv[2]);
  }
  printf("msq: %.4f\n", msq);

  double lambda = 10.0; // 0.1; // 20.0; // 1.0;
  if (argc > 3) {
    lambda = std::stod(argv[3]);
  }
  printf("lambda: %.4f\n", lambda);

  int l_max = 12;

  QfeLatticeS2 lattice(q);
  // lattice.Refine2D(n_refine);
  lattice.LoopRefine(6);
  lattice.Inflate();
  lattice.UpdateDistinct();
  lattice.UpdateAntipodes();
  lattice.UpdateWeights();
  lattice.UpdateYlm(l_max);

  QfePhi4 field(&lattice, msq, lambda);
  field.HotStart();
  field.metropolis_z = 0.1;

  // // apply exact counter terms
  // double ct_sum = 0.0;
  // for (int s = 0; s < lattice.n_sites; s++) {
  //   double log_wt = log(lattice.sites[s].wt);
  //   field.msq_ct[s] = log_wt * 4.135;
  //   ct_sum += field.msq_ct[s] * lattice.sites[s].wt;
  // }
  //
  // // subtract position-independent piece
  // double ct_mean = ct_sum / double(lattice.n_sites);
  // for (int s = 0; s < lattice.n_sites; s++) {
  //   field.msq_ct[s] -= ct_mean;
  // }

  // open the counter term file
  char path[50];
  sprintf(path, "ct/ct_%d_%d_loop.dat", q, n_refine);
  FILE* file = fopen(path, "r");
  if (file == nullptr) {
    fprintf(stderr, "unable to open counterterm file: %s\n", path);
  }

  // read the counter terms
  std::vector<double> ct(lattice.n_distinct);
  for (int i = 0; i < lattice.n_distinct; i++) {
    fscanf(file, "%lf", &ct[i]);
  }
  fclose(file);

  // apply the counter terms to each site
  for (int s = 0; s < lattice.n_sites; s++) {
    int id = lattice.sites[s].id;
    field.msq_ct[s] += -6.0 * field.lambda * ct[id];
  }

  // get spherical harmonic combinations that mix with the A irrep of I
  std::vector<double> C0(lattice.n_sites, real(lattice.ylm[0][0]));
  std::vector<double> C6(lattice.n_sites);
  std::vector<double> C10(lattice.n_sites);
  std::vector<double> C12(lattice.n_sites);
  for (int s = 0; s < lattice.n_sites; s++) {

    if (l_max >= 6) {
      Complex l6_0 = sqrt(11.0 / 25.0) * lattice.ylm[s][21];
      Complex l6_5 = 2.0 * sqrt(7.0 / 25.0) * lattice.ylm[s][26];
      C6[s] = real(l6_0 - l6_5);
    }

    if (l_max >= 10) {
      Complex l10_0 = sqrt(247.0 / 1875.0) * lattice.ylm[s][55];
      Complex l10_5 = 2.0 * sqrt(209.0 / 625.0) * lattice.ylm[s][60];
      Complex l10_10 = 2.0 * sqrt(187.0 / 1875.0) * lattice.ylm[s][65];
      C10[s] = real(l10_0 + l10_5 + l10_10);
    }

    if (l_max >= 12) {
      Complex l12_0 = sqrt(1071.0 / 3125.0) * lattice.ylm[s][78];
      Complex l12_5 = 2.0 * sqrt(286.0 / 3125.0) * lattice.ylm[s][83];
      Complex l12_10 = 2.0 * sqrt(741.0 / 3125.0) * lattice.ylm[s][88];
      C12[s] = real(l12_0 - l12_5 + l12_10);
    }
  }

  printf("initial action: %.12f\n", field.Action());

  // measurements
  std::vector<double> mag;
  std::vector<double> action;
  std::vector<QfeMeasReal> distinct_phi2(lattice.n_distinct);
  std::vector<QfeMeasReal> distinct_antipodal_phi2(lattice.n_distinct);
  QfeMeasReal phi2;
  QfeMeasReal antipodal_phi2;
  QfeMeasReal Q6;
  QfeMeasReal Q10;
  QfeMeasReal Q12;
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

    double phi2_sum = 0.0;
    double antipodal_phi2_sum = 0.0;
    double Q6_sum = 0.0;
    double Q10_sum = 0.0;
    double Q12_sum = 0.0;
    std::vector<double> distinct_phi2_sum(lattice.n_distinct);
    std::vector<double> distinct_antipodal_phi2_sum(lattice.n_distinct);
    for (int s = 0; s < lattice.n_sites; s++) {
      int a = lattice.antipode[s];
      double phi = field.phi[s];
      double antipodal_phi = field.phi[a];
      double wt = lattice.sites[s].wt;
      phi2_sum += phi * phi * wt;
      antipodal_phi2_sum += phi * antipodal_phi * wt;
      Q6_sum += C6[s] * phi * antipodal_phi * wt;
      Q10_sum += C10[s] * phi * antipodal_phi * wt;
      Q12_sum += C12[s] * phi * antipodal_phi * wt;

      // measure distinct sites
      int i = lattice.sites[s].id;
      distinct_phi2_sum[i] += phi * phi;
      distinct_antipodal_phi2_sum[i] += phi * antipodal_phi;
    }
    phi2.Measure(phi2_sum / double(lattice.n_sites));
    antipodal_phi2.Measure(antipodal_phi2_sum / double(lattice.n_sites));
    Q6.Measure(Q6_sum / double(lattice.n_sites));
    Q10.Measure(Q10_sum / double(lattice.n_sites));
    Q12.Measure(Q12_sum / double(lattice.n_sites));

    for (int i = 0; i < lattice.n_distinct; i++) {
      double n = double(lattice.distinct_n_sites[i]);
      distinct_phi2[i].Measure(distinct_phi2_sum[i] / n);
      distinct_antipodal_phi2[i].Measure(distinct_antipodal_phi2_sum[i] / n);
    }
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
  printf("U4: %.12e %.12e\n", U4(mag2, mag4), JackknifeU4(mag2, mag4));
  printf("susceptibility: %.12e (%.12e)\n", Susceptibility(mag2, mag_abs), \
      JackknifeSusceptibility(mag2, mag_abs));

  printf("phi2: %.12e (%.12e)\n", phi2.Mean(), phi2.Error());
  printf("antipodal_phi2: %.12e (%.12e)\n", \
      antipodal_phi2.Mean(), antipodal_phi2.Error());

  printf("Q6: %.12e (%.12e)\n", Q6.Mean(), Q6.Error());
  printf("Q10: %.12e (%.12e)\n", Q10.Mean(), Q10.Error());
  printf("Q12: %.12e (%.12e)\n", Q12.Mean(), Q12.Error());

  printf("distinct_phi2:\n");
  for (int i = 0; i < lattice.n_distinct; i++) {
    int s = lattice.distinct_first[i];
    printf("%04d %.12f %.12e %.12e\n", i, lattice.sites[s].wt, \
        distinct_phi2[i].Mean(), distinct_phi2[i].Error());
  }

  printf("distinct_antipodal_phi2:\n");
  for (int i = 0; i < lattice.n_distinct; i++) {
    int s = lattice.distinct_first[i];
    printf("%04d %.12f %.12e %.12e\n", i, lattice.sites[s].wt, \
        distinct_antipodal_phi2[i].Mean(), \
        distinct_antipodal_phi2[i].Error());
  }

  return 0;
}
