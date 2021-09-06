// ising_s2_crit.cc

#include <getopt.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>
#include "ising.h"
#include "s2.h"
#include "statistics.h"

typedef std::complex<double> Complex;

int main(int argc, char* argv[]) {

  // default parameters
  int n_refine = 8;
  int q = 5;
  double beta = 0.41;
  double ct_mult = 0.0;
  int n_therm = 2000;
  int n_traj = 100000;
  int n_skip = 10;
  int n_wolff = 5;
  int n_metropolis = 4;
  const struct option long_options[] = {
    { "n_refine", required_argument, 0, 'N' },
    { "q", required_argument, 0, 'q' },
    { "beta", required_argument, 0, 'b' },
    { "ct_mult", required_argument, 0, 'c' },
    { "n_therm", required_argument, 0, 'h' },
    { "n_traj", required_argument, 0, 't' },
    { "n_skip", required_argument, 0, 's' },
    { "n_wolff", required_argument, 0, 'w' },
    { "n_metropolis", required_argument, 0, 'e' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "N:q:b:c:h:t:s:w:e:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N': n_refine = atoi(optarg); break;
      case 'q': q = atoi(optarg); break;
      case 'b': beta = std::stod(optarg); break;
      case 'c': ct_mult = std::stod(optarg); break;
      case 'h': n_therm = atoi(optarg); break;
      case 't': n_traj = atoi(optarg); break;
      case 's': n_skip = atoi(optarg); break;
      case 'w': n_wolff = atoi(optarg); break;
      case 'e': n_metropolis = atoi(optarg); break;
      default: break;
    }
  }

  printf("n_therm: %d\n", n_therm);
  printf("n_traj: %d\n", n_traj);
  printf("n_skip: %d\n", n_skip);
  printf("n_wolff: %d\n", n_wolff);
  printf("n_metropolis: %d\n", n_metropolis);

  QfeLatticeS2 lattice(q);
  lattice.Refine2D(n_refine);
  lattice.Inflate();
  lattice.UpdateWeights();
  lattice.UpdateDistinct();
  lattice.UpdateAntipodes();
  lattice.UpdateYlm(12);
  printf("n_refine: %d\n", n_refine);
  printf("q: %d\n", q);
  printf("total sites: %d\n", lattice.n_sites);

  QfeIsing field(&lattice, beta);
  field.HotStart();
  printf("beta: %.4f\n", field.beta);
  printf("initial action: %.12f\n", field.Action());

  printf("ct_mult: %.12f\n", ct_mult);
  for (int l = 0; l < lattice.n_links; l++) {
    field.beta_ct[l] = ct_mult * lattice.links[l].wt;
  }

  // get spherical harmonic combinations that mix with the A irrep of I
  std::vector<double> C6(lattice.n_sites);
  std::vector<double> C10(lattice.n_sites);
  std::vector<double> C12(lattice.n_sites);
  for (int s = 0; s < lattice.n_sites; s++) {

    Complex l6_0 = sqrt(11.0 / 25.0) * lattice.ylm[s][21];
    Complex l6_5 = 2.0 * sqrt(7.0 / 25.0) * lattice.ylm[s][26];
    C6[s] = real(l6_0 - l6_5);

    Complex l10_0 = sqrt(247.0 / 1875.0) * lattice.ylm[s][55];
    Complex l10_5 = 2.0 * sqrt(209.0 / 625.0) * lattice.ylm[s][60];
    Complex l10_10 = 2.0 * sqrt(187.0 / 1875.0) * lattice.ylm[s][65];
    C10[s] = real(l10_0 + l10_5 + l10_10);

    Complex l12_0 = sqrt(1071.0 / 3125.0) * lattice.ylm[s][78];
    Complex l12_5 = 2.0 * sqrt(286.0 / 3125.0) * lattice.ylm[s][83];
    Complex l12_10 = 2.0 * sqrt(741.0 / 3125.0) * lattice.ylm[s][88];
    C12[s] = real(l12_0 - l12_5 + l12_10);
  }

  // measurements
  std::vector<double> spin;  // average spin
  std::vector<double> anti_spin;  // average antipodal spin
  std::vector<double> action;
  std::vector<QfeMeasReal> distinct_spin(lattice.n_distinct);
  std::vector<QfeMeasReal> distinct_anti_spin(lattice.n_distinct);
  QfeMeasReal Q6;
  QfeMeasReal Q10;
  QfeMeasReal Q12;
  QfeMeasReal Q6_anti;
  QfeMeasReal Q10_anti;
  QfeMeasReal Q12_anti;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;
  QfeMeasReal accept_overrelax;

  for (int n = 0; n < (n_traj + n_therm); n++) {

    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      cluster_size_sum += field.WolffUpdate();
    }
    double metropolis_sum = 0.0;
    for (int j = 0; j < n_metropolis; j++) {
      metropolis_sum += field.Metropolis();
    }
    cluster_size.Measure(double(cluster_size_sum) / double(lattice.n_sites));
    accept_metropolis.Measure(metropolis_sum);

    if (n % n_skip || n < n_therm) continue;

    // measure field
    double spin_sum = 0.0;
    double anti_spin_sum = 0.0;
    double Q6_sum = 0.0;
    double Q10_sum = 0.0;
    double Q12_sum = 0.0;
    double Q6_anti_sum = 0.0;
    double Q10_anti_sum = 0.0;
    double Q12_anti_sum = 0.0;
    std::vector<double> distinct_spin_sum(lattice.n_distinct, 0.0);
    std::vector<double> distinct_anti_spin_sum(lattice.n_distinct, 0.0);

    for (int s = 0; s < lattice.n_sites; s++) {
      int a = lattice.antipode[s];
      double spin_s = field.spin[s];
      double anti_spin_s = field.spin[a];
      double wt = lattice.sites[s].wt;

      spin_sum += spin_s * wt;
      anti_spin_sum += spin_s * anti_spin_s * wt;
      Q6_sum += C6[s] * spin_s * wt;
      Q10_sum += C10[s] * spin_s * wt;
      Q12_sum += C12[s] * spin_s * wt;
      Q6_anti_sum += C6[s] * spin_s * anti_spin_s * wt;
      Q10_anti_sum += C10[s] * spin_s * anti_spin_s * wt;
      Q12_anti_sum += C12[s] * spin_s * anti_spin_s * wt;

      // measure distinct sites
      int i = lattice.sites[s].id;
      distinct_spin_sum[i] += spin_s;
      distinct_anti_spin_sum[i] += spin_s * anti_spin_s;
    }
    spin.push_back(spin_sum / double(lattice.n_sites));
    anti_spin.push_back(anti_spin_sum / double(lattice.n_sites));
    Q6.Measure(Q6_sum / double(lattice.n_sites));
    Q10.Measure(Q10_sum / double(lattice.n_sites));
    Q12.Measure(Q12_sum / double(lattice.n_sites));
    Q6_anti.Measure(Q6_anti_sum / double(lattice.n_sites));
    Q10_anti.Measure(Q10_anti_sum / double(lattice.n_sites));
    Q12_anti.Measure(Q12_anti_sum / double(lattice.n_sites));

    for (int i = 0; i < lattice.n_distinct; i++) {
      double n = double(lattice.distinct_n_sites[i]);
      distinct_spin[i].Measure(distinct_spin_sum[i] / n);
      distinct_anti_spin[i].Measure(distinct_anti_spin_sum[i] / n);
    }

    action.push_back(field.Action());
    printf("%06d %.12f %.4f %.4f\n", \
        n, action.back(), \
        accept_metropolis.last, \
        cluster_size.last);
  }

  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());

  std::vector<double> mag_abs(spin.size());
  std::vector<double> mag2(spin.size());
  std::vector<double> mag4(spin.size());
  for (int i = 0; i < spin.size(); i++) {
    double m = spin[i];
    double m2 = m * m;
    mag_abs[i] = fabs(m);
    mag2[i] = m2;
    mag4[i] = m2 * m2;
  }

  printf("spin: %+.12e (%.12e), %.4f\n", \
      Mean(spin), JackknifeMean(spin), \
      AutocorrTime(spin));
  printf("anti_spin: %.12e (%.12e), %.4f\n", \
      Mean(anti_spin), JackknifeMean(anti_spin), \
      AutocorrTime(anti_spin));

  printf("m: %+.12e (%.12e), %.4f\n", \
      Mean(spin), JackknifeMean(spin), \
      AutocorrTime(spin));
  printf("m^2: %.12e (%.12e), %.4f\n", \
      Mean(mag2), JackknifeMean(mag2), \
      AutocorrTime(mag2));
  printf("m^4: %.12e (%.12e), %.4f\n", \
      Mean(mag4), JackknifeMean(mag4), \
      AutocorrTime(mag4));
  printf("U4: %.12e (%.12e)\n", \
      U4(mag2, mag4), \
      JackknifeU4(mag2, mag4));
  printf("m_susc: %.12e (%.12e)\n", \
      Susceptibility(mag2, mag_abs), \
      JackknifeSusceptibility(mag2, mag_abs));

  printf("Q6: %.12e (%.12e)\n", Q6.Mean(), Q6.Error());
  printf("Q10: %.12e (%.12e)\n", Q10.Mean(), Q10.Error());
  printf("Q12: %.12e (%.12e)\n", Q12.Mean(), Q12.Error());
  printf("Q6_anti: %.12e (%.12e)\n", Q6_anti.Mean(), Q6_anti.Error());
  printf("Q10_anti: %.12e (%.12e)\n", Q10_anti.Mean(), Q10_anti.Error());
  printf("Q12_anti: %.12e (%.12e)\n", Q12_anti.Mean(), Q12_anti.Error());

  printf("\n");
  for (int i = 0; i < lattice.n_distinct; i++) {
    int s = lattice.distinct_first[i];
    printf("%04d %3d %.12f %.12e %.12e %.12e %.12e\n", i, \
        lattice.distinct_n_sites[i], \
        lattice.sites[s].wt, \
        distinct_spin[i].Mean(), \
        distinct_spin[i].Error(), \
        distinct_anti_spin[i].Mean(), \
        distinct_anti_spin[i].Error());
  }

  FILE* out_file = fopen("ising_s2_crit.dat", "a");
  fprintf(out_file, "%d", n_refine);
  fprintf(out_file, " %d", q);
  fprintf(out_file, " %d", lattice.n_sites);
  fprintf(out_file, " %.12f", field.beta);
  fprintf(out_file, " %.12f", ct_mult);
  fprintf(out_file, " %+.12e %.12e", Mean(spin), JackknifeMean(spin));
  fprintf(out_file, " %.12e %.12e", Mean(anti_spin), JackknifeMean(anti_spin));
  fprintf(out_file, " %.12e %.12e", Mean(mag2), JackknifeMean(mag2));
  fprintf(out_file, " %.12e %.12e", Mean(mag_abs), JackknifeMean(mag_abs));
  fprintf(out_file, " %.12e %.12e", U4(mag2, mag4), JackknifeU4(mag2, mag4));
  fprintf(out_file, " %.12e %.12e", \
      Susceptibility(mag2, mag_abs), JackknifeSusceptibility(mag2, mag_abs));
  fprintf(out_file, " %.12e %.12e", Q6.Mean(), Q6.Error());
  fprintf(out_file, " %.12e %.12e", Q10.Mean(), Q10.Error());
  fprintf(out_file, " %.12e %.12e", Q12.Mean(), Q12.Error());
  fprintf(out_file, " %.12e %.12e", Q6_anti.Mean(), Q6_anti.Error());
  fprintf(out_file, " %.12e %.12e", Q10_anti.Mean(), Q10_anti.Error());
  fprintf(out_file, " %.12e %.12e", Q12_anti.Mean(), Q12_anti.Error());
  fprintf(out_file, "\n");
  fclose(out_file);

  char path[50];
  sprintf(path, "distinct_%d_%d_%04d.dat", n_refine, q, int(round(ct_mult * 1000)));
  out_file = fopen(path, "w");

  for (int i = 0; i < lattice.n_distinct; i++) {
    int s = lattice.distinct_first[i];
    fprintf(out_file, "%04d", i);
    fprintf(out_file, " %3d", lattice.distinct_n_sites[i]);
    fprintf(out_file, " %.12f", lattice.sites[s].wt);
    fprintf(out_file, " %.12e", distinct_spin[i].Mean());
    fprintf(out_file, " %.12e", distinct_spin[i].Error());
    fprintf(out_file, " %.12e", distinct_anti_spin[i].Mean());
    fprintf(out_file, " %.12e", distinct_anti_spin[i].Error());
    fprintf(out_file, "\n");
  }
  fclose(out_file);

  return 0;
}
