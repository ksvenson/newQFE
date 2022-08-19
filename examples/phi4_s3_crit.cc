// phi4_s2_crit.cc

#include <getopt.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "phi4.h"
#include "s3.h"
#include "statistics.h"

int main(int argc, char* argv[]) {

  // default parameters
  double msq = -1.82241 * 2;
  double lambda = 1.0;
  unsigned int seed = 1234u;
  bool cold_start = false;
  int n_therm = 2000;
  int n_traj = 20000;
  int n_skip = 10;
  int n_wolff = 5;
  int n_metropolis = 4;
  double metropolis_z = 0.1;
  std::string base_path = "../s3_refine/s3_std/q5k1";
  std::string data_dir = "phi4_s3_crit/q5k1";

  const struct option long_options[] = {
    { "seed", required_argument, 0, 'S' },
    { "cold_start", no_argument, 0, 'C' },
    { "msq", required_argument, 0, 'm' },
    { "lambda", required_argument, 0, 'L' },
    { "n_therm", required_argument, 0, 'h' },
    { "n_traj", required_argument, 0, 't' },
    { "n_skip", required_argument, 0, 's' },
    { "n_wolff", required_argument, 0, 'w' },
    { "n_metropolis", required_argument, 0, 'e' },
    { "metropolis_z", required_argument, 0, 'z' },
    { "base_path", required_argument, 0, 'b' },
    { "data_dir", required_argument, 0, 'd' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "S:Cm:L:h:t:s:w:e:z:b:d:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'S': seed = atol(optarg); break;
      case 'C': cold_start = true; break;
      case 'm': msq = std::stod(optarg); break;
      case 'L': lambda = std::stod(optarg); break;
      case 'h': n_therm = atoi(optarg); break;
      case 't': n_traj = atoi(optarg); break;
      case 's': n_skip = atoi(optarg); break;
      case 'w': n_wolff = atoi(optarg); break;
      case 'e': n_metropolis = atoi(optarg); break;
      case 'z': metropolis_z = std::stod(optarg); break;
      case 'b': base_path = optarg; break;
      case 'd': data_dir = optarg; break;
      default: break;
    }
  }

  printf("n_therm: %d\n", n_therm);
  printf("n_traj: %d\n", n_traj);
  printf("n_skip: %d\n", n_skip);
  printf("n_wolff: %d\n", n_wolff);
  printf("n_metropolis: %d\n", n_metropolis);

  QfeLatticeS3 lattice(0);
  char lattice_path[50];
  sprintf(lattice_path, "%s_lattice.dat", base_path.c_str());
  printf("opening lattice file: %s\n", lattice_path);
  FILE* file = fopen(lattice_path, "r");
  assert(file != nullptr);
  lattice.ReadLattice(file);
  fclose(file);

  lattice.SeedRng(seed);
  printf("total sites: %d\n", lattice.n_sites);

  lattice.vol = double(lattice.n_sites);

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

  // calculate ricci curvature term
  std::vector<double> ricci_scalar(lattice.n_distinct);
  for (int id = 0; id < lattice.n_distinct; id++) {
    int s_i = lattice.distinct_first[id];
    Eigen::Vector4d r_ric = Eigen::Vector4d::Zero();
    for (int n = 0; n < lattice.sites[s_i].nn; n++) {
      int l = lattice.sites[s_i].links[n];
      int s_j = lattice.sites[s_i].neighbors[n];
      r_ric += lattice.links[l].wt * (lattice.r[s_i] - lattice.r[s_j]);
    }
    ricci_scalar[id] = 0.5 * r_ric.norm() / lattice.sites[s_i].wt;
    printf("%04d %.12f\n", id, ricci_scalar[id]);
  }
  exit(0);

  // apply ricci term to all sites
  for (int s = 0; s < lattice.n_sites; s++) {
    int id = lattice.sites[s].id;
    // field.msq_ct[s] = ricci_scalar[id] / 6.0;  // = 1 / 4 R^2
  }

  // measurements
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
    cluster_size.Measure(double(cluster_size_sum) / lattice.vol);
    accept_metropolis.Measure(metropolis_sum);

    if (n % n_skip || n < n_therm) continue;

    // measure correlators
    double mag_sum = 0.0;

    for (int s = 0; s < lattice.n_sites; s++) {
      mag_sum += lattice.sites[s].wt * field.phi[s];
    }

    // measure magnetization
    double m = mag_sum / lattice.vol;
    double m_sq = m * m;
    mag.Measure(fabs(m));
    mag_2.Measure(m_sq);
    mag_4.Measure(m_sq * m_sq);
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
  // char run_id[50];
  // sprintf(run_id, "%d_%d", q, n_refine);

  // sprintf(path, "%s/%s/%s_%08X.dat", \
  //     data_dir.c_str(), run_id, run_id, seed);
  // printf("opening file: %s\n", path);
  // FILE* data_file = fopen(path, "w");
  // assert(data_file != nullptr);

  printf("action: %+.12e %.12e %.4f %.4f\n", \
      action.Mean(), action.Error(), \
      action.AutocorrFront(), action.AutocorrBack());
  // fprintf(data_file, "action %.16e %.16e %d\n", \
  //     action.Mean(), action.Error(), \
  //     action.n);
  printf("mag: %.12e %.12e %.4f %.4f\n", \
      m_mean, m_err, mag.AutocorrFront(), mag.AutocorrBack());
  // fprintf(data_file, "mag %.16e %.16e %d\n", \
  //     m_mean, m_err, mag.n);
  printf("m^2: %.12e %.12e %.4f %.4f\n", \
      m2_mean, m2_err, mag_2.AutocorrFront(), mag_2.AutocorrBack());
  // fprintf(data_file, "mag^2 %.16e %.16e %d\n", \
  //     m2_mean, m2_err, mag_2.n);
  printf("m^4: %.12e %.12e %.4f %.4f\n", \
      m4_mean, m4_err, \
      mag_4.AutocorrFront(), mag_4.AutocorrBack());
  // fprintf(data_file, "mag^4 %.16e %.16e %d\n", \
  //     m4_mean, m4_err, mag_4.n);

  double U4_mean = 1.5 * (1.0 - m4_mean / (3.0 * m2_mean * m2_mean));
  double U4_err = 0.5 * U4_mean * sqrt(pow(m4_err / m4_mean, 2.0) \
      + pow(2.0 * m2_err / m2_mean, 2.0));
  printf("U4: %.12e %.12e\n", U4_mean, U4_err);

  double m_susc_mean = (m2_mean - m_mean * m_mean) * lattice.vol;
  double m_susc_err = sqrt(pow(m2_err, 2.0) \
      + pow(2.0 * m_mean * m_err, 2.0)) * lattice.vol;
  printf("m_susc: %.12e %.12e\n", m_susc_mean, m_susc_err);

  return 0;
}
