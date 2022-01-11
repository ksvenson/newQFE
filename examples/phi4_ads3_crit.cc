// phi4_ads3_crit.cc

#include <getopt.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "ads3.h"
#include "phi4.h"
#include "statistics.h"

int main(int argc, char* argv[]) {

  // default parameters
  int n_layers = 3;
  int q = 7;
  int Nt = 0;  // default to number of boundary sites
  double msq = -1.0;
  double lambda = 1.0;
  int n_therm = 1000;
  int n_traj = 20000;
  int n_skip = 20;
  int n_wolff = 4;
  int n_metropolis = 2;
  double metropolis_z = 0.1;

  const struct option long_options[] = {
    { "n_layers", required_argument, 0, 'N' },
    { "q", required_argument, 0, 'q' },
    { "n_t", required_argument, 0, 'T' },
    { "msq", required_argument, 0, 'm' },
    { "lambda", required_argument, 0, 'l' },
    { "n_therm", required_argument, 0, 'h' },
    { "n_traj", required_argument, 0, 't' },
    { "n_skip", required_argument, 0, 's' },
    { "n_wolff", required_argument, 0, 'w' },
    { "n_metropolis", required_argument, 0, 'e' },
    { "metropolis_z", required_argument, 0, 'z' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "N:q:T:m:l:h:t:s:w:e:z:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N': n_layers = atoi(optarg); break;
      case 'q': q = atoi(optarg); break;
      case 'T': Nt = atoi(optarg); break;
      case 'm': msq = std::stod(optarg); break;
      case 'l': lambda = std::stod(optarg); break;
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

  QfeLatticeAdS3 lattice(n_layers, q, Nt);
  printf("n_layers: %d\n", lattice.n_layers);
  printf("q: %d\n", lattice.q);
  printf("Nt: %d\n", lattice.Nt);
  printf("total sites: %d\n", lattice.n_sites);
  printf("bulk sites: %d\n", lattice.n_bulk);
  printf("boundary sites: %d\n", lattice.n_boundary);
  printf("t_scale: %.12f\n", lattice.t_scale);

  QfePhi4 field(&lattice, msq, lambda);
  field.metropolis_z = metropolis_z;
  field.ColdStart();
  printf("msq: %.4f\n", field.msq);
  printf("lambda: %.4f\n", field.lambda);
  printf("metropolis_z: %.4f\n", field.metropolis_z);
  printf("initial action: %.12f\n", field.Action());

  // set up dirichlet boundary conditions
  for (int i = 0; i < lattice.n_boundary; i++) {
    int s = lattice.boundary_sites[i];
    field.phi[s] = 0.0;
    field.is_fixed[s] = true;
  }

  // calculate the lattice volume (include bulk sites only)
  lattice.vol = 0.0;
  for (int i = 0; i < lattice.n_bulk; i++) {
    int s = lattice.bulk_sites[i];
    lattice.vol += lattice.sites[s].wt;
  }

  // measurements
  QfeMeasReal phi;  // average phi (magnetization)
  QfeMeasReal phi2;  // average phi^2
  QfeMeasReal phi_abs;  // average abs(phi)
  QfeMeasReal mag;  // magnetization = abs( sum_i phi_i )
  QfeMeasReal mag_2;  // magnetization^2
  QfeMeasReal mag_4;  // magnetization^4
  QfeMeasReal action;
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
    cluster_size.Measure(double(cluster_size_sum) / double(lattice.n_bulk));
    accept_metropolis.Measure(metropolis_sum);
    accept_overrelax.Measure(field.Overrelax());

    if (n % n_skip || n < n_therm) continue;

    demon.Measure(field.overrelax_demon);

    // measurements
    double phi_sum = 0.0;
    double phi2_sum = 0.0;
    double phi_abs_sum = 0.0;
    for (int i = 0; i < lattice.n_bulk; i++) {
      int s = lattice.bulk_sites[i];
      double p1 = field.phi[s];
      phi_sum += p1 * lattice.sites[s].wt;
      phi2_sum += p1 * p1 * lattice.sites[s].wt;
      phi_abs_sum += fabs(p1) * lattice.sites[s].wt;
    }
    phi.Measure(phi_sum / lattice.vol);
    phi2.Measure(phi2_sum / lattice.vol);
    phi_abs.Measure(phi_abs_sum / lattice.vol);
    mag.Measure(fabs(phi_sum));
    mag_2.Measure(phi_sum * phi_sum);
    mag_4.Measure(mag_2.last * mag_2.last);

    action.Measure(field.Action());
    printf("%06d %.12f %.4f %.4f %.12f %.4f\n", \
        n, action.last, \
        accept_metropolis.last, \
        accept_overrelax.last, demon.last, \
        cluster_size.last);
  }

  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());
  printf("accept_overrelax: %.4f\n", accept_overrelax.Mean());
  printf("demon: %.12f (%.12f)\n", demon.Mean(), demon.Error());
  printf("action: %+.12e %.12e %.4f %.4f\n", \
      action.Mean(), action.Error(), \
      action.AutocorrFront(), action.AutocorrBack());

  double phi_abs_mean = phi_abs.Mean();
  double phi_abs_err = phi_abs.Error();
  double phi2_mean = phi2.Mean();
  double phi2_err = phi2.Error();

  printf("phi: %+.12e %.12e %.4f %.4f\n", \
      phi.Mean(), phi.Error(), \
      phi.AutocorrFront(), phi.AutocorrBack());
  printf("phi^2: %+.12e %.12e %.4f %.4f\n", \
      phi2.Mean(), phi2.Error(), \
      phi2.AutocorrFront(), phi2.AutocorrBack());
  printf("phi_abs: %+.12e %.12e %.4f %.4f\n", \
      phi_abs_mean, phi_abs_err, \
      phi_abs.AutocorrFront(), phi_abs.AutocorrBack());

  double phi_susc_mean = phi2_mean - phi_abs_mean * phi_abs_mean;
  double phi_susc_err = sqrt(pow(phi2_err, 2.0) + pow(2.0 * phi_abs_mean * phi_abs_err, 2.0));
  printf("phi_susc: %.12e %.12e\n", phi_susc_mean, phi_susc_err);

  double m_mean = mag.Mean();
  double m_err = mag.Error();
  double m2_mean = mag_2.Mean();
  double m2_err = mag_2.Error();
  double m4_mean = mag_4.Mean();
  double m4_err = mag_4.Error();

  printf("mag: %+.12e %.12e %.4f %.4f\n", \
      m_mean, m_err, \
      mag.AutocorrFront(), mag.AutocorrBack());
  printf("mag^2: %+.12e %.12e %.4f %.4f\n", \
      m2_mean, m2_err, \
      mag_2.AutocorrFront(), mag_2.AutocorrBack());
  printf("mag^4: %+.12e %.12e %.4f %.4f\n", \
      m4_mean, m4_err, \
      mag_4.AutocorrFront(), mag_4.AutocorrBack());

  double u4_mean = 1.5 * (1.0 - m4_mean / (3.0 * m2_mean * m2_mean));
  double u4_err = 0.5 * u4_mean * sqrt(pow(m4_err / m4_mean, 2.0) \
      + pow(2.0 * m2_err / m2_mean, 2.0));
  printf("U4: %.12e %.12e\n", u4_mean, u4_err);

  double m_susc_mean = (m2_mean - m_mean * m_mean) / lattice.vol;
  double m_susc_err = sqrt(pow(m2_err, 2.0) \
      + pow(2.0 * m_mean * m_err, 2.0)) / lattice.vol;
  printf("m_susc: %.12e %.12e\n", m_susc_mean, m_susc_err);

  FILE* file;

  file = fopen("phi4_ads3_crit.dat", "a");
  fprintf(file, "%d", lattice.n_layers);  // [0]
  fprintf(file, " %.12f", lattice.vol);  // [1]
  fprintf(file, " %.12f", field.msq);  // [2]
  fprintf(file, " %.4f", field.lambda);  // [3]
  fprintf(file, " %+.12e %.12e", phi.Mean(), phi.Error());  // [4,5]
  fprintf(file, " %.12e %.12e", phi2_mean, phi2_err);  // [6,7]
  fprintf(file, " %.12e %.12e", phi_abs_mean, phi_abs_err);  // [8,9]
  fprintf(file, " %.12e %.12e", phi_susc_mean, phi_susc_mean);  // [10,11]
  fprintf(file, " %.12e %.12e", m2_mean, m2_err);  // [12,13]
  fprintf(file, " %.12e %.12e", m_mean, m_err);  // [14,15]
  fprintf(file, " %.12e %.12e", u4_mean, u4_err);  // [16,17]
  fprintf(file, " %.12e %.12e", m_susc_mean, m_susc_err);  // [18,19]
  fprintf(file, "\n");
  fclose(file);

  return 0;
}
