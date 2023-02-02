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
#include "timer.h"
#include "util.h"

int main(int argc, char* argv[]) {
  // default parameters
  int n_refine = 8;
  int q = 5;
  double msq = -2.0;
  double lambda = 1.0;
  unsigned int seed = 1234u;
  bool cold_start = false;
  int l_max = 6;
  int n_therm = 2000;
  int n_traj = 20000;
  int n_skip = 10;
  int n_wolff = 5;
  int n_metropolis = 4;
  double metropolis_z = 0.1;
  bool do_overrelax = false;
  double wall_time = 0.0;
  std::string data_dir = "phi4_s2_crit";
  std::string orbit_path = "";

  const struct option long_options[] = {
      {"n_refine", required_argument, 0, 'N'},
      {"q", required_argument, 0, 'q'},
      {"seed", required_argument, 0, 'S'},
      {"cold_start", no_argument, 0, 'C'},
      {"msq", required_argument, 0, 'm'},
      {"lambda", required_argument, 0, 'L'},
      {"l_max", required_argument, 0, 'l'},
      {"n_therm", required_argument, 0, 'h'},
      {"n_traj", required_argument, 0, 't'},
      {"n_skip", required_argument, 0, 's'},
      {"n_wolff", required_argument, 0, 'w'},
      {"n_metropolis", required_argument, 0, 'e'},
      {"metropolis_z", required_argument, 0, 'z'},
      {"do_overrelax", no_argument, 0, 'o'},
      {"data_dir", required_argument, 0, 'd'},
      {"wall_time", required_argument, 0, 'W'},
      {"orbit_path", required_argument, 0, 'O'},
      {0, 0, 0, 0}};

  const char* short_options = "N:q:S:Cm:L:l:h:t:s:w:e:z:od:W:O:";

  while (true) {
    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N':
        n_refine = atoi(optarg);
        break;
      case 'q':
        q = atoi(optarg);
        break;
      case 'S':
        seed = atol(optarg);
        break;
      case 'C':
        cold_start = true;
        break;
      case 'm':
        msq = std::stod(optarg);
        break;
      case 'L':
        lambda = std::stod(optarg);
        break;
      case 'l':
        l_max = atoi(optarg);
        break;
      case 'h':
        n_therm = atoi(optarg);
        break;
      case 't':
        n_traj = atoi(optarg);
        break;
      case 's':
        n_skip = atoi(optarg);
        break;
      case 'w':
        n_wolff = atoi(optarg);
        break;
      case 'e':
        n_metropolis = atoi(optarg);
        break;
      case 'z':
        metropolis_z = std::stod(optarg);
        break;
      case 'o':
        do_overrelax = true;
        break;
      case 'd':
        data_dir = optarg;
        break;
      case 'W':
        wall_time = std::stod(optarg);
        break;
      case 'O':
        orbit_path = optarg;
        break;
      default:
        break;
    }
  }

  std::string run_id = string_format("l%.4fm%.6f", lambda, fabs(msq));
  printf("run_id: %s\n", run_id.c_str());

  printf("n_therm: %d\n", n_therm);
  printf("n_traj: %d\n", n_traj);
  printf("n_skip: %d\n", n_skip);
  printf("n_wolff: %d\n", n_wolff);
  printf("n_metropolis: %d\n", n_metropolis);

  // number of spherical harmonics to measure
  int n_ylm = ((l_max + 1) * (l_max + 2)) / 2;
  printf("l_max: %d\n", l_max);
  printf("n_ylm: %d\n", n_ylm);

  QfeLatticeS2 lattice(q, n_refine);
  if (!orbit_path.empty()) {
    FILE* orbit_file = fopen(orbit_path.c_str(), "r");
    assert(orbit_file != nullptr);
    lattice.ReadOrbits(orbit_file);
    fclose(orbit_file);
  }
  lattice.SeedRng(seed);
  lattice.UpdateWeights();

  printf("n_refine: %d\n", n_refine);
  printf("q: %d\n", q);
  printf("total sites: %d\n", lattice.n_sites);

  double vol = lattice.vol;
  double vol_sq = vol * vol;

  double a_lat = lattice.CalcLatticeSpacing();
  double lambda0 = lambda * a_lat * a_lat;
  QfePhi4 field(&lattice, msq, lambda0);

  // check if there is an rng file
  std::string rng_path =
      string_format("%s/%s/%s_rng_%08X.dat", data_dir.c_str(), run_id.c_str(),
                    run_id.c_str(), seed);
  printf("opening file: %s\n", rng_path.c_str());
  FILE* rng_file = fopen(rng_path.c_str(), "r");
  if (rng_file != nullptr) {
    printf("loading rng state from file\n");
    lattice.rng.ReadRng(rng_file);
    fclose(rng_file);
  }

  // check if there is a field checkpoint file
  std::string field_path =
      string_format("%s/%s/%s_field_%08X.dat", data_dir.c_str(), run_id.c_str(),
                    run_id.c_str(), seed);
  printf("opening file: %s\n", field_path.c_str());
  FILE* field_file = fopen(field_path.c_str(), "rb");
  if (field_file != nullptr) {
    printf("loading field from checkpoint file\n");
    field.ReadField(field_file);
    fclose(field_file);
  } else if (cold_start) {
    printf("cold start\n");
    field.ColdStart();
  } else {
    printf("hot start\n");
    field.HotStart();
  }

  field.metropolis_z = metropolis_z;
  printf("msq: %.4f\n", field.msq);
  printf("lambda: %.4f\n", lambda);
  printf("lambda0: %.4f\n", field.lambda);
  printf("metropolis_z: %.4f\n", field.metropolis_z);
  printf("initial action: %.12f\n", field.Action());

  // calculate spherical harmonics at each site. In all of these loops, (y_l,
  // y_m) are the integer eigenvalues of the spherical harmonics and y_i is a
  // sequential index over all of eigenvalues up to l_max (excluding negative
  // y_m).
  Eigen::MatrixXcd ylm(lattice.n_sites, n_ylm);
  for (int s = 0; s < lattice.n_sites; s++) {
    for (int y_i = 0, y_l = 0, y_m = 0; y_i < n_ylm; y_i++) {
      ylm(s, y_i) = lattice.CalcYlm(s, y_l, y_m);

      y_m++;
      if (y_m > y_l) {
        y_l++;
        y_m = 0;
      }
    }
  }

  // measurements
  std::vector<QfeMeasReal> legendre_2pt(l_max + 1);
  std::vector<QfeMeasReal> ylm_2pt(n_ylm);
  QfeMeasReal mag;     // magnetization
  QfeMeasReal mag_2;   // magnetization^2
  QfeMeasReal mag_4;   // magnetization^4
  QfeMeasReal mag_6;   // magnetization^6
  QfeMeasReal mag_8;   // magnetization^8
  QfeMeasReal mag_10;  // magnetization^10
  QfeMeasReal mag_12;  // magnetization^12
  QfeMeasReal action;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;
  QfeMeasReal accept_overrelax;
  QfeMeasReal overrelax_demon;

  // read bulk measurements
  std::string bulk_path =
      string_format("%s/%s/%s_bulk_%08X.dat", data_dir.c_str(), run_id.c_str(),
                    run_id.c_str(), seed);
  printf("opening file: %s\n", bulk_path.c_str());
  FILE* bulk_file = fopen(bulk_path.c_str(), "r");
  if (bulk_file != nullptr) {
    while (!feof(bulk_file)) {
      char meas_name[40];
      fscanf(bulk_file, "%s ", meas_name);
      if (strcmp(meas_name, "action") == 0) {
        action.ReadMeasurement(bulk_file);
      } else if (strcmp(meas_name, "mag") == 0) {
        mag.ReadMeasurement(bulk_file);
      } else if (strcmp(meas_name, "mag") == 0) {
        mag.ReadMeasurement(bulk_file);
      } else if (strcmp(meas_name, "mag^2") == 0) {
        mag_2.ReadMeasurement(bulk_file);
      } else if (strcmp(meas_name, "mag^4") == 0) {
        mag_4.ReadMeasurement(bulk_file);
      } else if (strcmp(meas_name, "mag^6") == 0) {
        mag_6.ReadMeasurement(bulk_file);
      } else if (strcmp(meas_name, "mag^8") == 0) {
        mag_8.ReadMeasurement(bulk_file);
      } else if (strcmp(meas_name, "mag^10") == 0) {
        mag_10.ReadMeasurement(bulk_file);
      } else if (strcmp(meas_name, "mag^12") == 0) {
        mag_12.ReadMeasurement(bulk_file);
      } else {
        printf("unknown measurement: %s\n", meas_name);
      }
    }
    fclose(bulk_file);
  }

  // read 2-point function legendre coefficients
  std::string legendre_2pt_path =
      string_format("%s/%s/%s_legendre_2pt_%08X.dat", data_dir.c_str(),
                    run_id.c_str(), run_id.c_str(), seed);
  printf("opening file: %s\n", legendre_2pt_path.c_str());
  FILE* legendre_2pt_file = fopen(legendre_2pt_path.c_str(), "r");
  if (legendre_2pt_file != nullptr) {
    int l;
    while (!feof(legendre_2pt_file)) {
      fscanf(legendre_2pt_file, "%d ", &l);
      if (l > l_max) {
        fscanf(legendre_2pt_file, "%*[^\n]");
      } else {
        legendre_2pt[l].ReadMeasurement(legendre_2pt_file);
      }
      fscanf(legendre_2pt_file, "\n");
    }
    fclose(legendre_2pt_file);
  }

  // read 2-point function ylm coefficients
  std::string ylm_2pt_path =
      string_format("%s/%s/%s_ylm_2pt_%08X.dat", data_dir.c_str(),
                    run_id.c_str(), run_id.c_str(), seed);
  printf("opening file: %s\n", ylm_2pt_path.c_str());
  FILE* ylm_2pt_file = fopen(ylm_2pt_path.c_str(), "r");
  if (ylm_2pt_file != nullptr) {
    int i_ylm, l, m;
    while (!feof(ylm_2pt_file)) {
      fscanf(ylm_2pt_file, "%d %d %d ", &i_ylm, &l, &m);
      if (i_ylm >= n_ylm) {
        fscanf(ylm_2pt_file, "%*[^\n]");
      } else {
        ylm_2pt[i_ylm].ReadMeasurement(ylm_2pt_file);
      }
      fscanf(ylm_2pt_file, "\n");
    }
    fclose(ylm_2pt_file);
  }

  Timer timer;

  for (int n = 0; n < (n_traj + n_therm); n++) {
    if (wall_time > 0.0 && timer.Duration() > wall_time) break;

    double metropolis_sum = 0.0;
    for (int j = 0; j < n_metropolis; j++) {
      metropolis_sum += field.Metropolis();
    }
    accept_metropolis.Measure(metropolis_sum);

    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      cluster_size_sum += field.WolffUpdate();
    }
    cluster_size.Measure(double(cluster_size_sum) / vol);

    // do overrelaxation update
    if (do_overrelax) {
      accept_overrelax.Measure(field.Overrelax());
    }

    if (n % n_skip || n < n_therm) continue;

    // measure overrelaxation demon
    if (do_overrelax) {
      overrelax_demon.Measure(field.overrelax_demon);
    }

    // measure correlators
    std::vector<Complex> ylm_2pt_sum(n_ylm, 0.0);
    double mag_sum = 0.0;

    for (int s = 0; s < lattice.n_sites; s++) {
      double wt_2pt = field.phi[s] * lattice.sites[s].wt;

      mag_sum += wt_2pt;

      for (int ylm_i = 0; ylm_i < n_ylm; ylm_i++) {
        Complex y = ylm(s, ylm_i);
        ylm_2pt_sum[ylm_i] += y * wt_2pt;
      }
    }

    double legendre_2pt_sum = 0.0;
    for (int ylm_i = 0, l = 0, m = 0; ylm_i < n_ylm; ylm_i++) {
      ylm_2pt[ylm_i].Measure(std::norm(ylm_2pt_sum[ylm_i]) / vol_sq);

      legendre_2pt_sum += ylm_2pt[ylm_i].last * (m == 0 ? 1.0 : 2.0);

      m++;
      if (m > l) {
        double coeff = 4.0 * M_PI / double(2 * l + 1);
        legendre_2pt[l].Measure(legendre_2pt_sum * coeff);
        legendre_2pt_sum = 0.0;
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
    mag_6.Measure(mag_4.last * m_sq);
    mag_8.Measure(mag_6.last * m_sq);
    mag_10.Measure(mag_8.last * m_sq);
    mag_12.Measure(mag_10.last * m_sq);
    action.Measure(field.Action());
    // printf("%06d %.12f %.4f %.4f\n", n, action.last, accept_metropolis.last,
    //        cluster_size.last);
  }

  timer.Stop();
  printf("duration: %.6f\n", timer.Duration());

  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());
  if (do_overrelax) {
    printf("accept_overrelax: %.4f\n", accept_overrelax.Mean());
    printf("overrelax_demon: %.12f %.12f\n", overrelax_demon.Mean(),
           overrelax_demon.Error());
  }

  // write rng state to file
  printf("writing rng state to file: %s\n", rng_path.c_str());
  rng_file = fopen(rng_path.c_str(), "w");
  assert(rng_file != nullptr);
  lattice.rng.WriteRng(rng_file);
  fclose(rng_file);

  // write field configuration to file
  printf("writing field to file: %s\n", field_path.c_str());
  field_file = fopen(field_path.c_str(), "wb");
  assert(field_file != nullptr);
  field.WriteField(field_file);
  fclose(field_file);

  double m_mean = mag.Mean();
  double m_err = mag.Error();
  double m2_mean = mag_2.Mean();
  double m2_err = mag_2.Error();
  double m4_mean = mag_4.Mean();
  double m4_err = mag_4.Error();

  // open an output file
  printf("opening file: %s\n", bulk_path.c_str());
  bulk_file = fopen(bulk_path.c_str(), "w");
  assert(bulk_file != nullptr);

  printf("action: %+.12e %.12e %.4f %.4f\n", action.Mean(), action.Error(),
         action.AutocorrFront(), action.AutocorrBack());
  fprintf(bulk_file, "action ");
  action.WriteMeasurement(bulk_file);

  printf("mag: %.12e %.12e %.4f %.4f\n", m_mean, m_err, mag.AutocorrFront(),
         mag.AutocorrBack());
  fprintf(bulk_file, "mag ");
  mag.WriteMeasurement(bulk_file);

  printf("m^2: %.12e %.12e %.4f %.4f\n", m2_mean, m2_err, mag_2.AutocorrFront(),
         mag_2.AutocorrBack());
  fprintf(bulk_file, "mag^2 ");
  mag_2.WriteMeasurement(bulk_file);

  printf("m^4: %.12e %.12e %.4f %.4f\n", m4_mean, m4_err, mag_4.AutocorrFront(),
         mag_4.AutocorrBack());
  fprintf(bulk_file, "mag^4 ");
  mag_4.WriteMeasurement(bulk_file);

  printf("m^6: %.12e %.12e %.4f %.4f\n", mag_6.Mean(), mag_6.Error(),
         mag_6.AutocorrFront(), mag_6.AutocorrBack());
  fprintf(bulk_file, "mag^6 ");
  mag_6.WriteMeasurement(bulk_file);

  printf("m^8: %.12e %.12e %.4f %.4f\n", mag_8.Mean(), mag_8.Error(),
         mag_8.AutocorrFront(), mag_8.AutocorrBack());
  fprintf(bulk_file, "mag^8 ");
  mag_8.WriteMeasurement(bulk_file);

  printf("m^10: %.12e %.12e %.4f %.4f\n", mag_10.Mean(), mag_10.Error(),
         mag_10.AutocorrFront(), mag_10.AutocorrBack());
  fprintf(bulk_file, "mag^10 ");
  mag_10.WriteMeasurement(bulk_file);

  printf("m^12: %.12e %.12e %.4f %.4f\n", mag_12.Mean(), mag_12.Error(),
         mag_12.AutocorrFront(), mag_12.AutocorrBack());
  fprintf(bulk_file, "mag^12 ");
  mag_12.WriteMeasurement(bulk_file);

  double U4_mean = 1.5 * (1.0 - m4_mean / (3.0 * m2_mean * m2_mean));
  double U4_err =
      0.5 * U4_mean *
      sqrt(pow(m4_err / m4_mean, 2.0) + pow(2.0 * m2_err / m2_mean, 2.0));
  printf("U4: %.12e %.12e\n", U4_mean, U4_err);

  double m_susc_mean = (m2_mean - m_mean * m_mean) * vol;
  double m_susc_err =
      sqrt(pow(m2_err, 2.0) + pow(2.0 * m_mean * m_err, 2.0)) * vol;
  printf("m_susc: %.12e %.12e\n", m_susc_mean, m_susc_err);

  // print 2-point function legendre coefficients
  printf("opening file: %s\n", legendre_2pt_path.c_str());
  legendre_2pt_file = fopen(legendre_2pt_path.c_str(), "w");
  assert(legendre_2pt_file != nullptr);
  for (int l = 0; l <= l_max; l++) {
    fprintf(legendre_2pt_file, "%02d ", l);
    legendre_2pt[l].WriteMeasurement(legendre_2pt_file);
  }
  fclose(legendre_2pt_file);

  // print 2-point function spherical harmonic coefficients
  printf("opening file: %s\n", ylm_2pt_path.c_str());
  ylm_2pt_file = fopen(ylm_2pt_path.c_str(), "w");
  assert(ylm_2pt_file != nullptr);
  for (int ylm_i = 0, l = 0, m = 0; ylm_i < n_ylm; ylm_i++) {
    fprintf(ylm_2pt_file, "%04d %02d %02d ", ylm_i, l, m);
    ylm_2pt[ylm_i].WriteMeasurement(ylm_2pt_file);
    m++;
    if (m > l) {
      l++;
      m = 0;
    }
  }
  fclose(ylm_2pt_file);

  return 0;
}
