// phi4_s3_corr.cc

#include <getopt.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>
#include "phi4.h"
#include "s3.h"
#include "statistics.h"
#include "timer.h"

typedef std::complex<double> Complex;

int main(int argc, char* argv[]) {

  // default parameters
  double msq = -0.2;
  double lambda = 0.1;
  unsigned int seed = 1234u;
  bool cold_start = false;
  int j_max = 12;
  int n_therm = 2000;
  int n_traj = 20000;
  int n_skip = 20;
  int n_wolff = 20;
  int n_metropolis = 4;
  double metropolis_z = 1.0;
  bool do_overrelax = false;
  bool use_ricci = false;
  double wall_time = 0.0;
  std::string ct_path = "";
  std::string lattice_path = "../s3_refine/s3_std/q5k1_grid.dat";
  std::string data_dir = "phi4_s3_corr/q5k1";

  const struct option long_options[] = {
    { "seed", required_argument, 0, 'S' },
    { "cold_start", no_argument, 0, 'C' },
    { "msq", required_argument, 0, 'm' },
    { "lambda", required_argument, 0, 'L' },
    { "j_max", required_argument, 0, 'j' },
    { "n_therm", required_argument, 0, 'h' },
    { "n_traj", required_argument, 0, 't' },
    { "n_skip", required_argument, 0, 's' },
    { "n_wolff", required_argument, 0, 'w' },
    { "n_metropolis", required_argument, 0, 'e' },
    { "metropolis_z", required_argument, 0, 'z' },
    { "do_overrelax", no_argument, 0, 'o' },
    { "use_ricci", no_argument, 0, 'R' },
    { "ct_path", required_argument, 0, 'c' },
    { "lattice_path", required_argument, 0, 'p' },
    { "data_dir", required_argument, 0, 'd' },
    { "wall_time", required_argument, 0, 'W' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "S:Cm:L:j:h:t:s:w:e:z:oRc:p:d:W:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'S': seed = atol(optarg); break;
      case 'C': cold_start = true; break;
      case 'm': msq = std::stod(optarg); break;
      case 'L': lambda = std::stod(optarg); break;
      case 'j': j_max = atoi(optarg); break;
      case 'h': n_therm = atoi(optarg); break;
      case 't': n_traj = atoi(optarg); break;
      case 's': n_skip = atoi(optarg); break;
      case 'w': n_wolff = atoi(optarg); break;
      case 'e': n_metropolis = atoi(optarg); break;
      case 'z': metropolis_z = std::stod(optarg); break;
      case 'o': do_overrelax = true; break;
      case 'R': use_ricci = true; break;
      case 'c': ct_path = optarg; break;
      case 'p': lattice_path = optarg; break;
      case 'd': data_dir = optarg; break;
      case 'W': wall_time = std::stod(optarg); break;
      default: break;
    }
  }

  printf("n_therm: %d\n", n_therm);
  printf("n_traj: %d\n", n_traj);
  printf("n_skip: %d\n", n_skip);
  printf("n_wolff: %d\n", n_wolff);
  printf("n_metropolis: %d\n", n_metropolis);
  printf("do_overrelax: %s\n", do_overrelax ? "yes": "no");
  printf("use_ricci: %s\n", use_ricci ? "yes": "no");
  printf("wall_time: %f\n", wall_time);

  // number of hyperspherical harmonics to measure
  int n_yjlm = (j_max + 1) * (j_max + 2) * (j_max + 3) / 6;
  printf("j_max: %d\n", j_max);
  printf("n_yjlm: %d\n", n_yjlm);

  QfeLatticeS3 lattice(0);
  printf("opening lattice file: %s\n", lattice_path.c_str());
  FILE* file = fopen(lattice_path.c_str(), "r");
  assert(file != nullptr);
  lattice.ReadLattice(file);
  lattice.UpdateAntipodes();
  fclose(file);

  lattice.SeedRng(seed);
  printf("total sites: %d\n", lattice.n_sites);

  lattice.vol = double(lattice.n_sites);
  double vol = lattice.vol;
  double vol_sq = vol * vol;

  QfePhi4 field(&lattice, msq, lambda);
  if (cold_start) {
    printf("cold start\n");
    field.ColdStart();
  } else {
    printf("hot start\n");
    field.HotStart();
  }
  field.metropolis_z = metropolis_z;
  printf("msq: %.6f\n", field.msq);
  printf("lambda: %.4f\n", field.lambda);
  printf("metropolis_z: %.4f\n", field.metropolis_z);
  printf("initial action: %.12f\n", field.Action());

  if (use_ricci) {
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
      // printf("%04d %.12f\n", id, ricci_scalar[id] / 6.0);
    }

    // apply ricci term to all sites
    for (int s = 0; s < lattice.n_sites; s++) {
      int id = lattice.sites[s].id;
      field.msq_ct[s] = ricci_scalar[id] / 6.0;  // = 1 / 4 R^2
    }
  }

  // open the counter term file
  if (!ct_path.empty()) {
    printf("opening counterterm file: %s\n", ct_path.c_str());
    FILE* ct_file = fopen(ct_path.c_str(), "r");
    assert(ct_file != nullptr);

    // read the counter terms
    std::vector<double> ct(lattice.n_distinct);
    std::vector<double> ct3(lattice.n_distinct);
    for (int i = 0; i < lattice.n_distinct; i++) {
      fscanf(ct_file, "%lf %lf", &ct[i], &ct3[i]);
    }
    fclose(ct_file);

    // apply the counter terms to each site
    for (int s = 0; s < lattice.n_sites; s++) {
      int id = lattice.sites[s].id;
      field.msq_ct[s] += -12.0 * field.lambda * ct[id];
      field.msq_ct[s] += 48.0 * field.lambda * field.lambda * ct3[id];
    }
  }

  // calculate hyperspherical harmonics at each site
  // in all of these loops, (y_j, y_l, y_m) are the integer eigenvalues
  // of the hyperspherical harmonics and y_i is a sequential index over
  // all of eigenvalues up to j_max (excluding negative y_m).
  Eigen::MatrixXcd yjlm(lattice.n_sites, n_yjlm);
  for (int s = 0; s < lattice.n_sites; s++) {
    for (int y_i = 0, y_j = 0, y_l = 0, y_m = 0; y_i < n_yjlm; y_i++) {

      yjlm(s, y_i) = lattice.GetYjlm(s, y_j, y_l, y_m);

      y_m++;
      if (y_m > y_l) {
        y_l++;
        y_m = 0;
        if (y_l > y_j) {
          y_j++;
          y_l = 0;
        }
      }
    }
  }

  // measurements
  std::vector<QfeMeasReal> legendre_2pt(j_max + 1);
  std::vector<QfeMeasReal> legendre_4pt(j_max + 1);
  // std::vector<QfeMeasReal> gegenbauer_2pt(j_max + 1);
  std::vector<QfeMeasReal> yjlm_2pt(n_yjlm);
  std::vector<QfeMeasReal> yjlm_4pt(n_yjlm);
  QfeMeasReal anti_2pt;  // antipodal 2-point function
  QfeMeasReal mag;  // magnetization
  QfeMeasReal mag_2;  // magnetization^2
  QfeMeasReal mag_4;  // magnetization^4
  QfeMeasReal action;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;
  QfeMeasReal accept_overrelax;
  QfeMeasReal overrelax_demon;

  Timer timer;

  for (int n = 0; n < (n_traj + n_therm); n++) {

    if (wall_time > 0.0 && timer.Duration() > wall_time) break;

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

    if (do_overrelax) {
      accept_overrelax.Measure(field.Overrelax());
    }

    if (n % n_skip || n < n_therm) continue;

    if (do_overrelax) {
      overrelax_demon.Measure(field.overrelax_demon);
    }

    // measure correlators
    std::vector<Complex> yjlm_2pt_sum(n_yjlm, 0.0);
    std::vector<Complex> yjlm_4pt_sum(n_yjlm, 0.0);
    // std::vector<double> gegenbauer_2pt_sum(j_max + 1, 0.0);
    double mag_sum = 0.0;
    double anti_2pt_sum = 0.0;

    for (int s = 0; s < lattice.n_sites; s++) {
      int a = lattice.antipode[s];
      double wt_2pt = field.phi[s] * lattice.sites[s].wt;
      double wt_4pt = wt_2pt * field.phi[a];

      mag_sum += wt_2pt;
      anti_2pt_sum += wt_4pt;

      for (int y_i = 0; y_i < n_yjlm; y_i++) {
        Complex y = yjlm(s, y_i);
        yjlm_2pt_sum[y_i] += y * wt_2pt;
        yjlm_4pt_sum[y_i] += y * wt_4pt;
      }

      // for (int s1 = 0; s1 < lattice.n_sites; s1++) {
      //   double cos_theta = lattice.r[s].dot(lattice.r[s1]);
      //   if (s == s1) cos_theta = 1.0;
      //   if (a == s1) cos_theta = -1.0;
      //   for (int y_j = 0; y_j <= j_max; y_j++) {
      //     gegenbauer_2pt_sum[y_j] += 0.25 / pow(M_PI, 4.0) * boost::math::gegenbauer(y_j, 1.0, cos_theta) * wt_2pt * field.phi[s1] * lattice.sites[s1].wt;
      //   }
      // }
    }

    double legendre_2pt_sum = 0.0;
    double legendre_4pt_sum = 0.0;
    for (int y_i = 0, y_j = 0, y_l = 0, y_m = 0; y_i < n_yjlm; y_i++) {
      yjlm_2pt[y_i].Measure(std::norm(yjlm_2pt_sum[y_i]) / vol_sq);
      yjlm_4pt[y_i].Measure(std::norm(yjlm_4pt_sum[y_i]) / vol_sq);

      legendre_2pt_sum += yjlm_2pt[y_i].last * (y_m == 0 ? 1.0 : 2.0);
      legendre_4pt_sum += yjlm_4pt[y_i].last * (y_m == 0 ? 1.0 : 2.0);

      y_m++;
      if (y_m > y_l) {
        y_l++;
        y_m = 0;
        if (y_l > y_j) {
          // gegenbauer_2pt[y_j].Measure(gegenbauer_2pt_sum[y_j] / vol_sq);
          double coeff = 1.0 / (2.0 * M_PI * M_PI) / double(y_j + 1);
          legendre_2pt[y_j].Measure(legendre_2pt_sum * coeff);
          legendre_4pt[y_j].Measure(legendre_4pt_sum * coeff);
          legendre_2pt_sum = 0.0;
          legendre_4pt_sum = 0.0;
          y_j++;
          y_l = 0;
        }
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

  timer.Stop();
  printf("duration: %.6f\n", timer.Duration());

  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());
  if (do_overrelax) {
    printf("accept_overrelax: %.4f\n", accept_overrelax.Mean());
    printf("overrelax_demon: %.12f %.12f\n", \
        overrelax_demon.Mean(), overrelax_demon.Error());
  }

  double m_mean = mag.Mean();
  double m_err = mag.Error();
  double m2_mean = mag_2.Mean();
  double m2_err = mag_2.Error();
  double m4_mean = mag_4.Mean();
  double m4_err = mag_4.Error();

  // open an output file
  char run_id[200];
  sprintf(run_id, "l%.4fm%.6f", lambda, -msq);

  char data_path[200];
  sprintf(data_path, "%s/%s/%s_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", data_path);
  FILE* data_file = fopen(data_path, "w");
  assert(data_file != nullptr);

  printf("action: %+.12e %.12e %.4f %.4f\n", \
      action.Mean(), action.Error(), \
      action.AutocorrFront(), action.AutocorrBack());
  fprintf(data_file, "action %.16e %.16e %d\n", \
      action.Mean(), action.Error(), \
      action.n);
  printf("mag: %.12e %.12e %.4f %.4f\n", \
      m_mean, m_err, mag.AutocorrFront(), mag.AutocorrBack());
  fprintf(data_file, "mag %.16e %.16e %d\n", \
      m_mean, m_err, mag.n);
  printf("m^2: %.12e %.12e %.4f %.4f\n", \
      m2_mean, m2_err, mag_2.AutocorrFront(), mag_2.AutocorrBack());
  fprintf(data_file, "mag^2 %.16e %.16e %d\n", \
      m2_mean, m2_err, mag_2.n);
  printf("m^4: %.12e %.12e %.4f %.4f\n", \
      m4_mean, m4_err, \
      mag_4.AutocorrFront(), mag_4.AutocorrBack());
  fprintf(data_file, "mag^4 %.16e %.16e %d\n", \
      m4_mean, m4_err, mag_4.n);
  printf("anti_2pt: %.12e %.12e %.4f %.4f\n", \
      anti_2pt.Mean(), anti_2pt.Error(), \
      anti_2pt.AutocorrFront(), anti_2pt.AutocorrBack());
  fprintf(data_file, "anti_2pt %.16e %.16e %d\n", \
      anti_2pt.Mean(), anti_2pt.Error(), anti_2pt.n);

  double U4_mean = 1.5 * (1.0 - m4_mean / (3.0 * m2_mean * m2_mean));
  double U4_err = 0.5 * U4_mean * sqrt(pow(m4_err / m4_mean, 2.0) \
      + pow(2.0 * m2_err / m2_mean, 2.0));
  printf("U4: %.12e %.12e\n", U4_mean, U4_err);

  double m_susc_mean = (m2_mean - m_mean * m_mean) * lattice.vol;
  double m_susc_err = sqrt(pow(m2_err, 2.0) \
      + pow(2.0 * m_mean * m_err, 2.0)) * lattice.vol;
  printf("m_susc: %.12e %.12e\n", m_susc_mean, m_susc_err);

  fclose(data_file);

  // // print 2-point function gegenbauer coefficients
  // for (int j = 0; j <= j_max; j++) {
  //   printf("gegenbauer_2pt_%02d:", j);
  //   printf(" %.12e", gegenbauer_2pt[j].Mean());
  //   printf(" %.12e", gegenbauer_2pt[j].Error());
  //   printf(" %.4f", gegenbauer_2pt[j].AutocorrFront());
  //   printf(" %.4f\n", gegenbauer_2pt[j].AutocorrBack());
  // }

  // print 2-point function legendre coefficients
  sprintf(data_path, "%s/%s/%s_legendre_2pt_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", data_path);
  data_file = fopen(data_path, "w");
  assert(data_file != nullptr);
  for (int j = 0; j <= j_max; j++) {
    printf("legendre_2pt_%02d:", j);
    printf(" %.12e", legendre_2pt[j].Mean());
    printf(" %.12e", legendre_2pt[j].Error());
    printf(" %.4f", legendre_2pt[j].AutocorrFront());
    printf(" %.4f\n", legendre_2pt[j].AutocorrBack());
    fprintf(data_file, "%04d %.16e %.16e %d\n", j, \
        legendre_2pt[j].Mean(), legendre_2pt[j].Error(), \
        legendre_2pt[j].n);
  }
  fclose(data_file);

  // print 4-point function legendre coefficients
  sprintf(data_path, "%s/%s/%s_legendre_4pt_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", data_path);
  data_file = fopen(data_path, "w");
  assert(data_file != nullptr);
  for (int j = 0; j <= j_max; j++) {
    printf("legendre_4pt_%02d:", j);
    printf(" %.12e", legendre_4pt[j].Mean());
    printf(" %.12e", legendre_4pt[j].Error());
    printf(" %.4f", legendre_4pt[j].AutocorrFront());
    printf(" %.4f\n", legendre_4pt[j].AutocorrBack());
    fprintf(data_file, "%04d %.16e %.16e %d\n", j, \
        legendre_4pt[j].Mean(), legendre_4pt[j].Error(), \
        legendre_4pt[j].n);
  }
  fclose(data_file);

  // print 2-point function spherical harmonic coefficients
  sprintf(data_path, "%s/%s/%s_yjlm_2pt_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", data_path);
  data_file = fopen(data_path, "w");
  assert(data_file != nullptr);
  for (int y_i = 0, y_j = 0, y_l = 0, y_m = 0; y_i < n_yjlm; y_i++) {
    printf("yjlm_2pt_%02d_%02d_%02d:", y_j, y_l, y_m);
    printf(" %.12e", yjlm_2pt[y_i].Mean());
    printf(" %.12e", yjlm_2pt[y_i].Error());
    printf(" %.4f", yjlm_2pt[y_i].AutocorrFront());
    printf(" %.4f\n", yjlm_2pt[y_i].AutocorrBack());
    fprintf(data_file, "%04d %.16e %.16e %d\n", y_i, \
        yjlm_2pt[y_i].Mean(), yjlm_2pt[y_i].Error(), \
        yjlm_2pt[y_i].n);
    y_m++;
    if (y_m > y_l) {
      y_l++;
      y_m = 0;
      if (y_l > y_j) {
        y_j++;
        y_l = 0;
      }
    }
  }
  fclose(data_file);

  // print 4-point function spherical harmonic coefficients
  sprintf(data_path, "%s/%s/%s_yjlm_4pt_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", data_path);
  data_file = fopen(data_path, "w");
  assert(data_file != nullptr);
  for (int y_i = 0, y_j = 0, y_l = 0, y_m = 0; y_i < n_yjlm; y_i++) {
    printf("yjlm_4pt_%02d_%02d_%02d:", y_j, y_l, y_m);
    printf(" %.12e", yjlm_4pt[y_i].Mean());
    printf(" %.12e", yjlm_4pt[y_i].Error());
    printf(" %.4f", yjlm_4pt[y_i].AutocorrFront());
    printf(" %.4f\n", yjlm_4pt[y_i].AutocorrBack());
    fprintf(data_file, "%04d %.16e %.16e %d\n", y_i, \
        yjlm_4pt[y_i].Mean(), yjlm_4pt[y_i].Error(), \
        yjlm_4pt[y_i].n);
    y_m++;
    if (y_m > y_l) {
      y_l++;
      y_m = 0;
      if (y_l > y_j) {
        y_j++;
        y_l = 0;
      }
    }
  }
  fclose(data_file);

  return 0;
}
