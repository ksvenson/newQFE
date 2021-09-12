// phi4_s2_crit.cc

#include <getopt.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>
#include "phi4.h"
#include "s2.h"
#include "statistics.h"

typedef std::complex<double> Complex;

int main(int argc, char* argv[]) {

  // default parameters
  int n_refine = 8;
  int q = 5;
  double msq = -21.96;
  double lambda = 10.0;
  double ct_mult = 0.0;
  const char* ct_dir = "./ct";
  double tri2_wt = 0.0;
  double tri3_wt = 0.0;
  double tri4_wt = 0.0;
  double tri5_wt = 0.0;
  int n_therm = 2000;
  int n_traj = 100000;
  int n_skip = 20;
  int n_wolff = 4;
  int n_metropolis = 1;
  double metropolis_z = 0.1;

  const struct option long_options[] = {
    { "n_refine", required_argument, 0, 'N' },
    { "q", required_argument, 0, 'q' },
    { "msq", required_argument, 0, 'm' },
    { "lambda", required_argument, 0, 'l' },
    { "ct_mult", required_argument, 0, 'c' },
    { "ct_dir", required_argument, 0, 'd' },
    { "tri2_wt", required_argument, 0, '2' },
    { "tri3_wt", required_argument, 0, '3' },
    { "tri4_wt", required_argument, 0, '4' },
    { "tri5_wt", required_argument, 0, '5' },
    { "n_therm", required_argument, 0, 'h' },
    { "n_traj", required_argument, 0, 't' },
    { "n_skip", required_argument, 0, 's' },
    { "n_wolff", required_argument, 0, 'w' },
    { "n_metropolis", required_argument, 0, 'e' },
    { "metropolis_z", required_argument, 0, 'z' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "N:q:m:l:c:d:2:3:4:5:h:t:s:w:e:z:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N': n_refine = atoi(optarg); break;
      case 'q': q = atoi(optarg); break;
      case 'm': msq = std::stod(optarg); break;
      case 'l': lambda = std::stod(optarg); break;
      case 'c': ct_mult = std::stod(optarg); break;
      case 'd': ct_dir = optarg; break;
      case '2': tri2_wt = std::stod(optarg); break;
      case '3': tri3_wt = std::stod(optarg); break;
      case '4': tri4_wt = std::stod(optarg); break;
      case '5': tri5_wt = std::stod(optarg); break;
      case 'h': n_therm = atoi(optarg); break;
      case 't': n_traj = atoi(optarg); break;
      case 's': n_skip = atoi(optarg); break;
      case 'w': n_wolff = atoi(optarg); break;
      case 'e': n_metropolis = atoi(optarg); break;
      case 'z': metropolis_z = std::stod(optarg); break;
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
  lattice.UpdateTriangleCoordinates();
  lattice.Inflate();
  lattice.UpdateWeights();
  lattice.UpdateDistinct();
  lattice.UpdateAntipodes();
  lattice.UpdateYlm(12);
  printf("n_refine: %d\n", n_refine);
  printf("q: %d\n", q);
  printf("total sites: %d\n", lattice.n_sites);

  QfePhi4 field(&lattice, msq, lambda);
  field.HotStart();
  field.metropolis_z = metropolis_z;
  printf("msq: %.4f\n", field.msq);
  printf("lambda: %.4f\n", field.lambda);
  printf("metropolis_z: %.4f\n", field.metropolis_z);
  printf("initial action: %.12f\n", field.Action());

  // open the counter term file
  char path[50];
  sprintf(path, "%s/ct_%d_%d.dat", ct_dir, q, n_refine);
  FILE* ct_file = fopen(path, "r");
  if (ct_file == nullptr) {
    fprintf(stderr, "unable to open counterterm file: %s\n", path);
  }

  // read the counter terms
  std::vector<double> ct(lattice.n_distinct);
  for (int i = 0; i < lattice.n_distinct; i++) {
    fscanf(ct_file, "%lf", &ct[i]);
  }
  fclose(ct_file);

  // apply the counter terms to each site
  printf("ct_mult: %.12f\n", ct_mult);
  for (int s = 0; s < lattice.n_sites; s++) {
    int id = lattice.sites[s].id;
    field.msq_ct[s] += -12.0 * field.lambda * ct[id] * ct_mult;
  }

  // apply triangular coordinate counterterms
  printf("tri2_wt: %.12f\n", tri2_wt);
  printf("tri3_wt: %.12f\n", tri3_wt);
  printf("tri4_wt: %.12f\n", tri4_wt);
  printf("tri5_wt: %.12f\n", tri5_wt);
  double ct_sum = 0.0;
  for (int s = 0; s < lattice.n_sites; s++) {
    int id = lattice.sites[s].id;
    double tri2 = lattice.tri2[id];
    double tri3 = lattice.tri3[id];
    field.msq_ct[s] += tri2_wt * tri2;
    field.msq_ct[s] += tri3_wt * tri3;
    field.msq_ct[s] += tri4_wt * tri2 * tri2;
    field.msq_ct[s] += tri5_wt * tri2 * tri3;
    ct_sum += field.msq_ct[s] * lattice.sites[s].wt;
  }

  // subtract out the position-independent part of the counterterms
  double ct_mean = ct_sum / double(lattice.n_sites);
  for (int s = 0; s < lattice.n_sites; s++) {
    field.msq_ct[s] -= ct_mean;
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
  std::vector<double> phi;  // average phi
  std::vector<double> phi2;  // average phi^2
  std::vector<double> anti_phi2;  // average antipodal phi^2
  std::vector<double> phi_abs;  // average abs(phi)
  std::vector<double> action;
  std::vector<QfeMeasReal> distinct_phi2(lattice.n_distinct);
  std::vector<QfeMeasReal> distinct_anti_phi2(lattice.n_distinct);
  QfeMeasReal Q6;
  QfeMeasReal Q10;
  QfeMeasReal Q12;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;
  QfeMeasReal accept_overrelax;
  QfeMeasReal demon;

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
    accept_overrelax.Measure(field.Overrelax());

    if (n % n_skip || n < n_therm) continue;

    demon.Measure(field.overrelax_demon);

    // measure field
    double phi_sum = 0.0;
    double phi2_sum = 0.0;
    double anti_phi2_sum = 0.0;
    double phi_abs_sum = 0.0;
    double Q6_sum = 0.0;
    double Q10_sum = 0.0;
    double Q12_sum = 0.0;
    std::vector<double> distinct_phi2_sum(lattice.n_distinct, 0.0);
    std::vector<double> distinct_anti_phi2_sum(lattice.n_distinct, 0.0);

    for (int s = 0; s < lattice.n_sites; s++) {
      int a = lattice.antipode[s];
      double phi_s = field.phi[s];
      double anti_phi_s = field.phi[a];
      double wt = lattice.sites[s].wt;

      phi_sum += phi_s * wt;
      phi2_sum += phi_s * phi_s * wt;
      phi_abs_sum += fabs(phi_s) * wt;
      anti_phi2_sum += phi_s * anti_phi_s * wt;
      Q6_sum += C6[s] * phi_s * anti_phi_s * wt;
      Q10_sum += C10[s] * phi_s * anti_phi_s * wt;
      Q12_sum += C12[s] * phi_s * anti_phi_s * wt;

      // measure distinct sites
      int i = lattice.sites[s].id;
      distinct_phi2_sum[i] += phi_s * phi_s;
      distinct_anti_phi2_sum[i] += phi_s * anti_phi_s;
    }
    phi.push_back(phi_sum / double(lattice.n_sites));
    phi2.push_back(phi2_sum / double(lattice.n_sites));
    anti_phi2.push_back(anti_phi2_sum / double(lattice.n_sites));
    phi_abs.push_back(phi_abs_sum / double(lattice.n_sites));
    Q6.Measure(Q6_sum / double(lattice.n_sites));
    Q10.Measure(Q10_sum / double(lattice.n_sites));
    Q12.Measure(Q12_sum / double(lattice.n_sites));

    for (int i = 0; i < lattice.n_distinct; i++) {
      double n = double(lattice.distinct_n_sites[i]);
      distinct_phi2[i].Measure(distinct_phi2_sum[i] / n);
      distinct_anti_phi2[i].Measure(distinct_anti_phi2_sum[i] / n);
    }

    action.push_back(field.Action());
    printf("%06d %.12f %.4f %.4f %.12f %.4f\n", \
        n, action.back(), \
        accept_metropolis.last, \
        accept_overrelax.last, demon.last, \
        cluster_size.last);
  }

  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());
  printf("accept_overrelax: %.4f\n", accept_overrelax.Mean());
  printf("demon: %.12f (%.12f)\n", demon.Mean(), demon.Error());

  std::vector<double> mag_abs(phi.size());
  std::vector<double> mag2(phi.size());
  std::vector<double> mag4(phi.size());
  for (int i = 0; i < phi.size(); i++) {
    double m = phi[i];
    double m2 = m * m;
    mag_abs[i] = fabs(m);
    mag2[i] = m2;
    mag4[i] = m2 * m2;
  }

  printf("phi: %+.12e (%.12e), %.4f\n", \
      Mean(phi), JackknifeMean(phi), \
      AutocorrTime(phi));
  printf("phi^2: %.12e (%.12e), %.4f\n", \
      Mean(phi2), JackknifeMean(phi2), \
      AutocorrTime(phi2));
  printf("phi_abs: %.12e (%.12e), %.4f\n", \
      Mean(phi_abs), JackknifeMean(phi_abs), \
      AutocorrTime(phi_abs));
  printf("phi_susc: %.12e (%.12e)\n", \
      Susceptibility(phi2, phi_abs), \
      JackknifeSusceptibility(phi2, phi_abs));

  printf("m: %+.12e (%.12e), %.4f\n", \
      Mean(phi), JackknifeMean(phi), \
      AutocorrTime(phi));
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

  printf("\n");
  for (int i = 0; i < lattice.n_distinct; i++) {
    int s = lattice.distinct_first[i];
    printf("%04d %3d %.12f %.12e %.12e %.12e %.12e\n", i, \
        lattice.distinct_n_sites[i], \
        lattice.sites[s].wt, \
        distinct_phi2[i].Mean(), \
        distinct_phi2[i].Error(), \
        distinct_anti_phi2[i].Mean(), \
        distinct_anti_phi2[i].Error());
  }

  FILE* out_file = fopen("phi4_s2_crit.dat", "a");
  fprintf(out_file, "%d", n_refine);
  fprintf(out_file, " %d", q);
  fprintf(out_file, " %d", lattice.n_sites);
  fprintf(out_file, " %.12f", field.msq);
  fprintf(out_file, " %.4f", field.lambda);
  fprintf(out_file, " %.12f", ct_mult);
  fprintf(out_file, " %+.12e %.12e", Mean(phi), JackknifeMean(phi));
  fprintf(out_file, " %.12e %.12e", Mean(phi2), JackknifeMean(phi2));
  fprintf(out_file, " %.12e %.12e", Mean(phi_abs), JackknifeMean(phi_abs));
  fprintf(out_file, " %.12e %.12e", \
      Susceptibility(phi2, phi_abs), JackknifeSusceptibility(phi2, phi_abs));
  fprintf(out_file, " %.12e %.12e", Mean(mag2), JackknifeMean(mag2));
  fprintf(out_file, " %.12e %.12e", Mean(mag_abs), JackknifeMean(mag_abs));
  fprintf(out_file, " %.12e %.12e", U4(mag2, mag4), JackknifeU4(mag2, mag4));
  fprintf(out_file, " %.12e %.12e", \
      Susceptibility(mag2, mag_abs), JackknifeSusceptibility(mag2, mag_abs));
  fprintf(out_file, " %.12e %.12e", Q6.Mean(), Q6.Error());
  fprintf(out_file, " %.12e %.12e", Q10.Mean(), Q10.Error());
  fprintf(out_file, " %.12e %.12e", Q12.Mean(), Q12.Error());
  fprintf(out_file, " %.4f", tri2_wt);
  fprintf(out_file, " %.4f", tri3_wt);
  fprintf(out_file, " %.4f", tri4_wt);
  fprintf(out_file, " %.4f", tri5_wt);
  fprintf(out_file, "\n");
  fclose(out_file);

  sprintf(path, "distinct_s%d_q%d_%04d_%04d_%04d_%04d.dat", \
      n_refine, q, \
      int(round(tri2_wt * 1000)), \
      int(round(tri3_wt * 1000)), \
      int(round(tri4_wt * 1000)), \
      int(round(tri5_wt * 1000)));
  out_file = fopen(path, "w");

  for (int i = 0; i < lattice.n_distinct; i++) {
    int s = lattice.distinct_first[i];
    fprintf(out_file, "%04d", i);
    fprintf(out_file, " %3d", lattice.distinct_n_sites[i]);
    fprintf(out_file, " %.12f", lattice.sites[s].wt);
    fprintf(out_file, " %.12e", distinct_phi2[i].Mean());
    fprintf(out_file, " %.12e", distinct_phi2[i].Error());
    fprintf(out_file, " %.12e", distinct_anti_phi2[i].Mean());
    fprintf(out_file, " %.12e", distinct_anti_phi2[i].Error());
    fprintf(out_file, "\n");
  }
  fclose(out_file);

  return 0;
}
