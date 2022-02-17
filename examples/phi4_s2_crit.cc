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
  double msq = -1.82241 * 2;
  double lambda = 1.0;
  const char* ct_dir = "./ct";
  unsigned int seed = 1234u;
  bool cold_start = false;
  int l_max = 6;
  int n_therm = 2000;
  int n_traj = 20000;
  int n_skip = 10;
  int n_wolff = 5;
  int n_metropolis = 4;
  double metropolis_z = 0.1;
  std::string data_dir = "phi4_s2_crit";

  const struct option long_options[] = {
    { "n_refine", required_argument, 0, 'N' },
    { "q", required_argument, 0, 'q' },
    { "seed", required_argument, 0, 'S' },
    { "cold_start", no_argument, 0, 'C' },
    { "msq", required_argument, 0, 'm' },
    { "lambda", required_argument, 0, 'L' },
    { "ct_dir", required_argument, 0, 'c' },
    { "l_max", required_argument, 0, 'l' },
    { "n_therm", required_argument, 0, 'h' },
    { "n_traj", required_argument, 0, 't' },
    { "n_skip", required_argument, 0, 's' },
    { "n_wolff", required_argument, 0, 'w' },
    { "n_metropolis", required_argument, 0, 'e' },
    { "metropolis_z", required_argument, 0, 'z' },
    { "data_dir", required_argument, 0, 'd' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "N:q:S:Cm:L:c:l:h:t:s:w:e:z:d:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N': n_refine = atoi(optarg); break;
      case 'q': q = atoi(optarg); break;
      case 'S': seed = atol(optarg); break;
      case 'C': cold_start = true; break;
      case 'm': msq = std::stod(optarg); break;
      case 'L': lambda = std::stod(optarg); break;
      case 'c': ct_dir = optarg; break;
      case 'l': l_max = atoi(optarg); break;
      case 'h': n_therm = atoi(optarg); break;
      case 't': n_traj = atoi(optarg); break;
      case 's': n_skip = atoi(optarg); break;
      case 'w': n_wolff = atoi(optarg); break;
      case 'e': n_metropolis = atoi(optarg); break;
      case 'z': metropolis_z = std::stod(optarg); break;
      case 'd': data_dir = optarg; break;
      default: break;
    }
  }

  printf("n_therm: %d\n", n_therm);
  printf("n_traj: %d\n", n_traj);
  printf("n_skip: %d\n", n_skip);
  printf("n_wolff: %d\n", n_wolff);
  printf("n_metropolis: %d\n", n_metropolis);

  // number of spherical harmonics to measure
  int n_ylm = ((l_max + 1) * (l_max + 2)) / 2;
  printf("l_max: %d\n", l_max);
  printf("n_ylm: %d\n", n_ylm);

  QfeLatticeS2 lattice(q);
  lattice.SeedRng(seed);
  lattice.Refine2D(n_refine);
  lattice.Inflate();
  lattice.UpdateWeights();
  lattice.UpdateDistinct();
  lattice.UpdateAntipodes();
  lattice.UpdateYlm(l_max);
  printf("n_refine: %d\n", n_refine);
  printf("q: %d\n", q);
  printf("total sites: %d\n", lattice.n_sites);

  QfePhi4 field(&lattice, msq, lambda);
  if (cold_start) {
    printf("cold start\n");
    field.ColdStart();
  } else {
    printf("hot start\n");
    field.HotStart();
  }
  field.metropolis_z = metropolis_z;
  printf("msq: %.4f\n", field.msq);
  printf("lambda: %.4f\n", field.lambda);
  printf("metropolis_z: %.4f\n", field.metropolis_z);
  printf("initial action: %.12f\n", field.Action());

  double vol = lattice.vol;
  double vol_sq = vol * vol;

  // open the counter term file
  char path[200];
  sprintf(path, "%s/ct_%d_%d.dat", ct_dir, q, n_refine);
  FILE* ct_file = fopen(path, "r");
  assert(ct_file != nullptr);

  // read the counter terms
  std::vector<double> ct(lattice.n_distinct);
  double ct3_dummy;  // not used in 2d
  for (int i = 0; i < lattice.n_distinct; i++) {
    fscanf(ct_file, "%lf %lf", &ct[i], &ct3_dummy);
  }
  fclose(ct_file);

  // apply the counter terms to each site
  for (int s = 0; s < lattice.n_sites; s++) {
    int id = lattice.sites[s].id;
    field.msq_ct[s] += -12.0 * field.lambda * ct[id];
  }

  // measurements
  std::vector<QfeMeasReal> legendre_2pt(l_max + 1);
  std::vector<QfeMeasReal> legendre_4pt(l_max + 1);
  std::vector<QfeMeasReal> ylm_2pt(n_ylm);
  std::vector<QfeMeasReal> ylm_4pt(n_ylm);
  QfeMeasReal anti_2pt;  // antipodal 2-point function
  QfeMeasReal mag;  // magnetization
  QfeMeasReal mag_2;  // magnetization^2
  QfeMeasReal mag_4;  // magnetization^4
  QfeMeasReal action;
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
    cluster_size.Measure(double(cluster_size_sum) / vol);
    accept_metropolis.Measure(metropolis_sum);

    if (n % n_skip || n < n_therm) continue;

    // measure correlators
    std::vector<Complex> ylm_2pt_sum(n_ylm, 0.0);
    std::vector<Complex> ylm_4pt_sum(n_ylm, 0.0);
    double mag_sum = 0.0;
    double anti_2pt_sum = 0.0;

    for (int s = 0; s < lattice.n_sites; s++) {
      int a = lattice.antipode[s];
      double wt_2pt = field.phi[s] * lattice.sites[s].wt;
      double wt_4pt = wt_2pt * field.phi[a];

      mag_sum += wt_2pt;
      anti_2pt_sum += wt_4pt;

      for (int ylm_i = 0; ylm_i < n_ylm; ylm_i++) {
        Complex y = lattice.ylm[s][ylm_i];
        ylm_2pt_sum[ylm_i] += y * wt_2pt;
        ylm_4pt_sum[ylm_i] += y * wt_4pt;
      }
    }

    double legendre_2pt_sum = 0.0;
    double legendre_4pt_sum = 0.0;
    for (int ylm_i = 0, l = 0, m = 0; ylm_i < n_ylm; ylm_i++) {
      ylm_2pt[ylm_i].Measure(std::norm(ylm_2pt_sum[ylm_i]) / vol_sq);
      ylm_4pt[ylm_i].Measure(std::norm(ylm_4pt_sum[ylm_i]) / vol_sq);

      legendre_2pt_sum += ylm_2pt[ylm_i].last * (m == 0 ? 1.0 : 2.0);
      legendre_4pt_sum += ylm_4pt[ylm_i].last * (m == 0 ? 1.0 : 2.0);

      m++;
      if (m > l) {
        double coeff = 4.0 * M_PI / double(2 * l + 1);
        legendre_2pt[l].Measure(legendre_2pt_sum * coeff);
        legendre_4pt[l].Measure(legendre_4pt_sum * coeff);
        legendre_2pt_sum = 0.0;
        legendre_4pt_sum = 0.0;
        l++;
        m = 0;
      }
    }

    // measure magnetization
    double m = mag_sum / vol;
    double m_sq = m * m;
    mag.Measure(fabs(m));
    mag_2.Measure(m_sq);
    mag_4.Measure(m_sq * m_sq);
    anti_2pt.Measure(anti_2pt_sum / lattice.vol);
    action.Measure(field.Action());
    printf("%06d %.12f %.4f %.4f\n", \
        n, action.last, \
        accept_metropolis.last, \
        cluster_size.last);
  }

  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());

  double m_mean = mag.Mean();
  double m_err = mag.Error();
  double m2_mean = mag_2.Mean();
  double m2_err = mag_2.Error();
  double m4_mean = mag_4.Mean();
  double m4_err = mag_4.Error();

  // open an output file
  char run_id[50];
  sprintf(run_id, "%d_%d", q, n_refine);
  FILE* file;

  sprintf(path, "%s/%s/%s_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", path);
  file = fopen(path, "w");
  assert(file != nullptr);

  printf("action: %+.12e %.12e %.4f %.4f\n", \
      action.Mean(), action.Error(), \
      action.AutocorrFront(), action.AutocorrBack());
  fprintf(file, "action %.16e %.16e %d\n", \
      action.Mean(), action.Error(), \
      action.n);
  printf("mag: %.12e %.12e %.4f %.4f\n", \
      m_mean, m_err, mag.AutocorrFront(), mag.AutocorrBack());
  fprintf(file, "mag %.16e %.16e %d\n", \
      m_mean, m_err, mag.n);
  printf("m^2: %.12e %.12e %.4f %.4f\n", \
      m2_mean, m2_err, mag_2.AutocorrFront(), mag_2.AutocorrBack());
  fprintf(file, "mag^2 %.16e %.16e %d\n", \
      m2_mean, m2_err, mag_2.n);
  printf("m^4: %.12e %.12e %.4f %.4f\n", \
      m4_mean, m4_err, \
      mag_4.AutocorrFront(), mag_4.AutocorrBack());
  fprintf(file, "mag^4 %.16e %.16e %d\n", \
      m4_mean, m4_err, mag_4.n);
  printf("anti_2pt: %.12e %.12e %.4f %.4f\n", \
      anti_2pt.Mean(), anti_2pt.Error(), \
      anti_2pt.AutocorrFront(), anti_2pt.AutocorrBack());
  fprintf(file, "anti_2pt %.16e %.16e %d\n", \
      anti_2pt.Mean(), anti_2pt.Error(), anti_2pt.n);
  fclose(file);

  double U4_mean = 1.5 * (1.0 - m4_mean / (3.0 * m2_mean * m2_mean));
  double U4_err = 0.5 * U4_mean * sqrt(pow(m4_err / m4_mean, 2.0) \
      + pow(2.0 * m2_err / m2_mean, 2.0));
  printf("U4: %.12e %.12e\n", U4_mean, U4_err);

  double m_susc_mean = (m2_mean - m_mean * m_mean) * vol;
  double m_susc_err = sqrt(pow(m2_err, 2.0) \
      + pow(2.0 * m_mean * m_err, 2.0)) * vol;
  printf("m_susc: %.12e %.12e\n", m_susc_mean, m_susc_err);

  // print 2-point function legendre coefficients
  sprintf(path, "%s/%s/%s_legendre_2pt_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", path);
  file = fopen(path, "w");
  assert(file != nullptr);
  for (int l = 0; l <= l_max; l++) {
    printf("legendre_2pt_%02d: %.12e", l, legendre_2pt[l].Mean());
    printf(" %.12e", legendre_2pt[l].Error());
    printf(" %.4f", legendre_2pt[l].AutocorrFront());
    printf(" %.4f\n", legendre_2pt[l].AutocorrBack());
    fprintf(file, "%04d %.16e %.16e %d\n", l, \
        legendre_2pt[l].Mean(), legendre_2pt[l].Error(), \
        legendre_2pt[l].n);
  }
  fclose(file);

  // print 4-point function legendre coefficients
  sprintf(path, "%s/%s/%s_legendre_4pt_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", path);
  file = fopen(path, "w");
  assert(file != nullptr);
  for (int l = 0; l <= l_max; l++) {
    printf("legendre_4pt_%02d: %.12e", l, legendre_4pt[l].Mean());
    printf(" %.12e", legendre_4pt[l].Error());
    printf(" %.4f", legendre_4pt[l].AutocorrFront());
    printf(" %.4f\n", legendre_4pt[l].AutocorrBack());
    fprintf(file, "%04d %.16e %.16e %d\n", l, \
        legendre_4pt[l].Mean(), legendre_4pt[l].Error(), \
        legendre_4pt[l].n);
  }
  fclose(file);

  // print 2-point function spherical harmonic coefficients
  sprintf(path, "%s/%s/%s_ylm_2pt_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", path);
  file = fopen(path, "w");
  assert(file != nullptr);
  for (int ylm_i = 0, l = 0, m = 0; ylm_i < n_ylm; ylm_i++) {
    printf("ylm_2pt_%02d_%02d: %.12e", l, m, ylm_2pt[ylm_i].Mean());
    printf(" %.12e", ylm_2pt[ylm_i].Error());
    printf(" %.4f", ylm_2pt[ylm_i].AutocorrFront());
    printf(" %.4f\n", ylm_2pt[ylm_i].AutocorrBack());
    fprintf(file, "%04d %.16e %.16e %d\n", ylm_i, \
        ylm_2pt[ylm_i].Mean(), ylm_2pt[ylm_i].Error(), \
        ylm_2pt[ylm_i].n);
    m++;
    if (m > l) {
      l++;
      m = 0;
    }
  }
  fclose(file);

  // print 2-point function spherical harmonic coefficients
  sprintf(path, "%s/%s/%s_ylm_4pt_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", path);
  file = fopen(path, "w");
  assert(file != nullptr);
  for (int ylm_i = 0, l = 0, m = 0; ylm_i < n_ylm; ylm_i++) {
    printf("ylm_4pt_%02d_%02d: %.12e", l, m, ylm_4pt[ylm_i].Mean());
    printf(" %.12e", ylm_4pt[ylm_i].Error());
    printf(" %.4f", ylm_4pt[ylm_i].AutocorrFront());
    printf(" %.4f\n", ylm_4pt[ylm_i].AutocorrBack());
    fprintf(file, "%04d %.16e %.16e %d\n", ylm_i, \
        ylm_4pt[ylm_i].Mean(), ylm_4pt[ylm_i].Error(), \
        ylm_4pt[ylm_i].n);
    m++;
    if (m > l) {
      l++;
      m = 0;
    }
  }
  fclose(file);

  return 0;
}
