// ising_rad_crit.cc

#include <getopt.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ising.h"
#include "lattice.h"
#include "statistics.h"
#include "timer.h"
#include "util.h"

#define FORMAT_DATA_PATH(NAME)                                   \
  string_format("%s/%s/%s_" #NAME "_%08X.dat", data_dir.c_str(), \
                run_id.c_str(), run_id.c_str(), seed);

int main(int argc, char* argv[]) {
  // default parameters
  int n_x = 8;
  int n_t = 64;
  double t_ratio = 1.0;
  unsigned int seed = 1234u;
  bool cold_start = false;
  int m_max = 6;
  int n_therm = 2000;
  int n_traj = 20000;
  int n_skip = 10;
  int n_wolff = 5;
  int n_metropolis = 4;
  double wall_time = 0.0;
  std::string data_dir = "ising_rad_crit";

  const struct option long_options[] = {
      {"n_x", required_argument, 0, 'X'},
      {"n_t", required_argument, 0, 'T'},
      {"t_ratio", required_argument, 0, 'A'},
      {"seed", required_argument, 0, 'S'},
      {"cold_start", no_argument, 0, 'C'},
      {"m_max", required_argument, 0, 'M'},
      {"n_therm", required_argument, 0, 'h'},
      {"n_traj", required_argument, 0, 't'},
      {"n_skip", required_argument, 0, 's'},
      {"n_wolff", required_argument, 0, 'w'},
      {"n_metropolis", required_argument, 0, 'e'},
      {"data_dir", required_argument, 0, 'd'},
      {"wall_time", required_argument, 0, 'W'},
      {0, 0, 0, 0}};

  const char* short_options = "X:T:A:S:CM:h:t:s:w:e:d:W:";

  while (true) {
    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'X':
        n_x = atoi(optarg);
        break;
      case 'T':
        n_t = atoi(optarg);
        break;
      case 'A':
        t_ratio = std::stod(optarg);
        break;
      case 'S':
        seed = atol(optarg);
        break;
      case 'C':
        cold_start = true;
        break;
      case 'M':
        m_max = atoi(optarg);
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
      case 'd':
        data_dir = optarg;
        break;
      case 'W':
        wall_time = std::stod(optarg);
        break;
      default:
        break;
    }
  }

  std::string run_id = string_format("x%dt%da%.4f", n_x, n_t, t_ratio);
  printf("run_id: %s\n", run_id.c_str());

  printf("n_therm: %d\n", n_therm);
  printf("n_traj: %d\n", n_traj);
  printf("n_skip: %d\n", n_skip);
  printf("n_wolff: %d\n", n_wolff);
  printf("n_metropolis: %d\n", n_metropolis);

  double K_x = 0.5 * asinh(1.0 / t_ratio);
  double K_t = 0.5 * asinh(t_ratio);
  printf("K_x: %.12f\n", K_x);
  printf("K_t: %.12f\n", K_t);

  // create a refined triangular lattice
  QfeLattice lattice;
  lattice.InitRect(n_x, n_t, K_x, K_t);
  lattice.SeedRng(seed);

  printf("n_x: %d\n", n_x);
  printf("n_t: %d\n", n_t);
  printf("total sites: %d\n", lattice.n_sites);

  double vol = lattice.vol;
  double vol_sq = vol * vol;
  int t_half = n_t / 2;

  QfeIsing field(&lattice, 1.0);

  // generate data file paths (macro is defined above)
  std::string rng_path = FORMAT_DATA_PATH(rng);
  std::string field_path = FORMAT_DATA_PATH(field);
  std::string bulk_path = FORMAT_DATA_PATH(bulk);
  std::string spin_2pt_path = FORMAT_DATA_PATH(spin_2pt);
  std::string spin_4pt_path = FORMAT_DATA_PATH(spin_4pt);

  // check if there is an rng file
  printf("opening file: %s\n", rng_path.c_str());
  FILE* rng_file = fopen(rng_path.c_str(), "r");
  if (rng_file != nullptr) {
    printf("loading rng state from file\n");
    lattice.rng.ReadRng(rng_file);
    fclose(rng_file);
  }

  // check if there is a field checkpoint file
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

  printf("initial action: %.12f\n", field.Action());

  // measurements
  std::vector<std::vector<QfeMeasReal>> spin_2pt(m_max + 1);
  std::vector<std::vector<QfeMeasReal>> spin_4pt(m_max + 1);
  for (int m = 0; m <= m_max; m++) {
    spin_2pt[m].resize(t_half + 1);
    spin_4pt[m].resize(t_half + 1);
  }
  QfeMeasReal mag;     // magnetization
  QfeMeasReal mag_2;   // magnetization^2
  QfeMeasReal mag_4;   // magnetization^4
  QfeMeasReal mag_6;   // magnetization^6
  QfeMeasReal mag_8;   // magnetization^8
  QfeMeasReal mag_10;  // magnetization^10
  QfeMeasReal mag_12;  // magnetization^12
  QfeMeasReal anti_2pt;
  QfeMeasReal action;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;

  // read bulk measurements
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
      } else if (strcmp(meas_name, "anti_2pt") == 0) {
        anti_2pt.ReadMeasurement(bulk_file);
      } else {
        printf("unknown measurement: %s\n", meas_name);
      }
    }
    fclose(bulk_file);
  }

  // read 2-point function legendre coefficients
  FILE* spin_2pt_file = fopen(spin_2pt_path.c_str(), "r");
  if (spin_2pt_file != nullptr) {
    printf("reading measurements from file: %s\n", spin_2pt_path.c_str());
    int m, t;
    while (!feof(spin_2pt_file)) {
      fscanf(spin_2pt_file, "%d %d ", &m, &t);
      if (m > m_max || t > t_half) {
        fscanf(spin_2pt_file, "%*[^\n]");
      } else {
        spin_2pt[m][t].ReadMeasurement(spin_2pt_file);
      }
    }
    fclose(spin_2pt_file);
  }

  // read 4-point function coefficients
  FILE* spin_4pt_file = fopen(spin_4pt_path.c_str(), "r");
  if (spin_4pt_file != nullptr) {
    printf("reading measurements from file: %s\n", spin_4pt_path.c_str());
    int m, t;
    while (!feof(spin_4pt_file)) {
      fscanf(spin_4pt_file, "%d %d ", &m, &t);
      if (m > m_max || t > t_half) {
        fscanf(spin_4pt_file, "%*[^\n]");
      } else {
        spin_4pt[m][t].ReadMeasurement(spin_4pt_file);
      }
    }
    fclose(spin_4pt_file);
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

    if (n % n_skip || n < n_therm) continue;

    // measure correlators
    std::vector<std::vector<double>> spin_2pt_sum(m_max + 1);
    std::vector<std::vector<double>> spin_4pt_sum(m_max + 1);
    for (int m = 0; m <= m_max; m++) {
      spin_2pt_sum[m].resize(n_t, 0.0);
      spin_4pt_sum[m].resize(n_t, 0.0);
    }
    double spin_sum = 0.0;
    double anti_2pt_sum = 0.0;

    for (int s = 0; s < lattice.n_sites; s++) {
      int x = s % n_x;
      int t = s / n_x;

      // antipodal site
      int a = (x + n_x / 2) % n_x + t * n_x;

      double wt_2pt = field.spin[s];
      double wt_4pt = wt_2pt * field.spin[a];

      spin_sum += wt_2pt;
      anti_2pt_sum += wt_4pt;

      for (int m = 0; m <= m_max; m++) {
        double cos_m_theta = cos(2.0 * M_PI * double(m * x) / double(n_x));
        spin_2pt_sum[m][t] += wt_2pt * cos_m_theta;
        spin_4pt_sum[m][t] += wt_4pt * cos_m_theta;
      }
    }

    // measure coefficients of the 2-point and 4-point function
    std::vector<double> spin_2pt_partial(t_half + 1, 0.0);
    std::vector<double> spin_4pt_partial(t_half + 1, 0.0);

    for (int m = 0; m <= m_max; m++) {
      // sum over all pairs of time slices
      for (int t1 = 0; t1 < n_t; t1++) {
        for (int t2 = t1; t2 < n_t; t2++) {
          // this calculates the shortest distance between t1 and t2
          int dt = (n_t - abs(2 * abs(t1 - t2) - n_t)) / 2;

          // double counting factor for t == n_t / 2
          double t_mult = 1.0;
          if (dt == t_half) t_mult = 2.0;

          double y2 = spin_2pt_sum[m][t1] * spin_2pt_sum[m][t2];
          double y4 = spin_4pt_sum[m][t1] * spin_4pt_sum[m][t2];
          spin_2pt_partial[dt] += y2 * t_mult;
          spin_4pt_partial[dt] += y4 * t_mult;
        }
      }

      for (int t = 0; t <= t_half; t++) {
        spin_2pt[m][t].Measure(spin_2pt_partial[t] / vol_sq * double(n_t));
        spin_4pt[m][t].Measure(spin_4pt_partial[t] / vol_sq * double(n_t));
        spin_2pt_partial[t] = 0.0;
        spin_4pt_partial[t] = 0.0;
      }
    }

    double m = spin_sum / vol;
    double m_sq = m * m;
    mag.Measure(fabs(m));
    mag_2.Measure(m_sq);
    mag_4.Measure(m_sq * m_sq);
    mag_6.Measure(mag_4.last * m_sq);
    mag_8.Measure(mag_6.last * m_sq);
    mag_10.Measure(mag_8.last * m_sq);
    mag_12.Measure(mag_10.last * m_sq);
    anti_2pt.Measure(anti_2pt_sum / vol);
    action.Measure(field.Action());
    // printf("%06d %.12f %.4f %.4f\n", \
    //     n, action.last, \
    //     accept_metropolis.last, \
    //     cluster_size.last);
  }

  timer.Stop();
  printf("duration: %.6f\n", timer.Duration());

  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());

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

  printf("anti_2pt: %.12e %.12e %.4f %.4f\n", anti_2pt.Mean(), anti_2pt.Error(),
         anti_2pt.AutocorrFront(), anti_2pt.AutocorrBack());
  fprintf(bulk_file, "anti_2pt ");
  anti_2pt.WriteMeasurement(bulk_file);

  fclose(bulk_file);

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
  printf("writing measurements to file: %s\n", spin_2pt_path.c_str());
  spin_2pt_file = fopen(spin_2pt_path.c_str(), "w");
  assert(spin_2pt_file != nullptr);
  for (int m = 0; m <= m_max; m++) {
    for (int t = 0; t <= t_half; t++) {
      fprintf(spin_2pt_file, "%02d %04d ", m, t);
      spin_2pt[m][t].WriteMeasurement(spin_2pt_file);
    }
  }
  fclose(spin_2pt_file);

  // print 4-point function legendre coefficients
  printf("writing measurements to file: %s\n", spin_4pt_path.c_str());
  spin_4pt_file = fopen(spin_4pt_path.c_str(), "w");
  assert(spin_4pt_file != nullptr);
  for (int m = 0; m <= m_max; m++) {
    for (int t = 0; t <= t_half; t++) {
      fprintf(spin_4pt_file, "%02d %04d ", m, t);
      spin_4pt[m][t].WriteMeasurement(spin_4pt_file);
    }
  }
  fclose(spin_4pt_file);

  return 0;
}
