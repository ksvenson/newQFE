// ising_ads3_crit.cc

#include <getopt.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "ads3.h"
#include "ising.h"
#include "statistics.h"

int main(int argc, char* argv[]) {

  // default parameters
  int n_layers = 3;
  int q = 7;
  int Nt = 0;  // default to number of boundary sites
  double beta = 0.116;
  int n_therm = 2000;
  int n_traj = 20000;
  int n_skip = 20;
  int n_wolff = 2;
  int n_metropolis = 4;

  const struct option long_options[] = {
    { "n_layers", required_argument, 0, 'N' },
    { "q", required_argument, 0, 'q' },
    { "n_t", required_argument, 0, 'T' },
    { "beta", required_argument, 0, 'b' },
    { "n_therm", required_argument, 0, 'h' },
    { "n_traj", required_argument, 0, 't' },
    { "n_skip", required_argument, 0, 's' },
    { "n_wolff", required_argument, 0, 'w' },
    { "n_metropolis", required_argument, 0, 'e' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "N:q:T:b:h:t:s:w:e:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N': n_layers = atoi(optarg); break;
      case 'q': q = atoi(optarg); break;
      case 'T': Nt = atoi(optarg); break;
      case 'b': beta = std::stod(optarg); break;
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

  QfeLatticeAdS3 lattice(n_layers, q, Nt);
  printf("n_layers: %d\n", lattice.n_layers);
  printf("q: %d\n", lattice.q);
  printf("Nt: %d\n", lattice.Nt);
  printf("total sites: %d\n", lattice.n_sites);
  printf("bulk sites: %d\n", lattice.n_bulk);
  printf("boundary sites: %d\n", lattice.n_boundary);
  printf("t_scale: %.12f\n", lattice.t_scale);

  QfeIsing field(&lattice, beta);
  field.HotStart();
  printf("beta: %.4f\n", field.beta);
  printf("initial action: %.12f\n", field.Action());

  // measurements
  std::vector<double> spin;  // average spin (magnetization)
  std::vector<double> action;
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
    cluster_size.Measure(double(cluster_size_sum) / double(lattice.n_sites));
    accept_metropolis.Measure(metropolis_sum);

    if (n % n_skip || n < n_therm) continue;

    // spin measurements
    double spin_sum = 0.0;
    for (int s = 0; s < lattice.n_sites; s++) {
      spin_sum += field.spin[s] * lattice.sites[s].wt;
    }
    spin.push_back(spin_sum / double(lattice.n_sites));

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

  FILE* file;

  file = fopen("ads3_ising_crit.dat", "a");
  fprintf(file, "%d", lattice.n_layers);  // [0]
  fprintf(file, " %d", lattice.n_sites);  // [1]
  fprintf(file, " %.12f", field.beta);  // [2]
  fprintf(file, " %+.12e %.12e", Mean(spin), JackknifeMean(spin));  // [3,4]
  fprintf(file, " %.12e %.12e", Mean(mag2), JackknifeMean(mag2));  // [5,6]
  fprintf(file, " %.12e %.12e", Mean(mag_abs), JackknifeMean(mag_abs));  // [7,8]
  fprintf(file, " %.12e %.12e", U4(mag2, mag4), JackknifeU4(mag2, mag4));  // [9,10]
  fprintf(file, " %.12e", Susceptibility(mag2, mag_abs));  // [11]
  fprintf(file, " %.12e", JackknifeSusceptibility(mag2, mag_abs));  // [12]
  fprintf(file, "\n");
  fclose(file);

  return 0;
}
