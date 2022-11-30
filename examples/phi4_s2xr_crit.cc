// phi4_s2xr_crit.cc

#include <getopt.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

#include "phi4.h"
#include "s2.h"
#include "statistics.h"
#include "timer.h"

typedef std::complex<double> Complex;

template <typename... Args>
std::string string_format(const char* format, Args... args) {
  int size = std::snprintf(nullptr, 0, format, args...) + 1;
  assert(size > 0);
  char buf[size];
  sprintf(buf, format, args...);
  return std::string(buf);
}

#define FORMAT_DATA_PATH(NAME)                                   \
  string_format("%s/%s/%s_" #NAME "_%08X.dat", data_dir.c_str(), \
                run_id.c_str(), run_id.c_str(), seed);

int main(int argc, char* argv[]) {
  // default parameters
  int n_refine = 4;
  int n_t = 64;
  int q = 5;
  double msq = -0.2702 * 2.0;
  double lambda = 0.2;
  unsigned int seed = 1234u;
  bool cold_start = false;
  int l_max = 6;
  int n_therm = 2000;
  int n_traj = 20000;
  int n_skip = 20;
  int n_wolff = 5;
  int n_metropolis = 4;
  double metropolis_z = 1.0;
  bool do_overrelax = false;
  bool use_ricci = false;
  double wall_time = 0.0;
  std::string ct_path = "";       // path to counterterm file
  std::string lattice_path = "";  // path to lattice file
  std::string data_dir = "phi4_s2xr_crit";
  bool verbose = false;

  const struct option long_options[] = {
      {"n_refine", required_argument, 0, 'N'},
      {"n_t", required_argument, 0, 'T'},
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
      {"use_ricci", no_argument, 0, 'R'},
      {"wall_time", required_argument, 0, 'W'},
      {"ct_path", required_argument, 0, 'c'},
      {"lattice_path", required_argument, 0, 'P'},
      {"data_dir", required_argument, 0, 'd'},
      {"verbose", no_argument, 0, 'v'},
      {0, 0, 0, 0}};

  const char* short_options = "N:T:q:S:Cm:L:l:h:t:s:w:e:z:oRW:c:P:d:v";

  while (true) {
    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N':
        n_refine = atoi(optarg);
        break;
      case 'T':
        n_t = atoi(optarg);
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
      case 'R':
        use_ricci = true;
        break;
      case 'W':
        wall_time = std::stod(optarg);
        break;
      case 'c':
        ct_path = optarg;
        break;
      case 'P':
        lattice_path = optarg;
        break;
      case 'd':
        data_dir = optarg;
        break;
      case 'v':
        verbose = true;
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
  printf("do_overrelax: %s\n", do_overrelax ? "yes" : "no");
  printf("use_ricci: %s\n", use_ricci ? "yes" : "no");
  printf("wall_time: %f\n", wall_time);
  printf("verbose output: %s\n", verbose ? "yes" : "no");

  // number of spherical harmonics to measure
  int n_ylm = ((l_max + 1) * (l_max + 2)) / 2;
  printf("l_max: %d\n", l_max);
  printf("n_ylm: %d\n", n_ylm);

  QfeLatticeS2 lattice(q);
  if (!lattice_path.empty()) {
    printf("opening lattice file: %s\n", lattice_path.c_str());
    FILE* lattice_file = fopen(lattice_path.c_str(), "r");
    assert(lattice_file != nullptr);
    lattice.ReadLattice(lattice_file);
    fclose(lattice_file);
  } else {
    lattice.Refine2D(n_refine);
    lattice.Inflate();
    lattice.UpdateWeights();
    lattice.UpdateDistinct();
    printf("n_refine: %d\n", n_refine);
    printf("q: %d\n", q);
  }
  lattice.SeedRng(seed);
  lattice.UpdateAntipodes();
  lattice.UpdateYlm(l_max);
  int n_sites_slice = lattice.n_sites;
  lattice.AddDimension(n_t);
  lattice.vol = double(lattice.n_sites);

  printf("n_t: %d\n", n_t);
  printf("total sites: %d\n", lattice.n_sites);

  QfePhi4 field(&lattice, msq, lambda);
  field.metropolis_z = metropolis_z;

  // generate data file paths (macro is defined above)
  std::string rng_path = FORMAT_DATA_PATH(rng);
  std::string field_path = FORMAT_DATA_PATH(field);
  std::string bulk_path = FORMAT_DATA_PATH(bulk);
  std::string legendre_2pt_path = FORMAT_DATA_PATH(legendre_2pt);
  std::string legendre_4pt_path = FORMAT_DATA_PATH(legendre_4pt);
  std::string ylm_2pt_path = FORMAT_DATA_PATH(ylm_2pt);
  std::string ylm_4pt_path = FORMAT_DATA_PATH(ylm_4pt);

  // check if there is an rng file
  FILE* rng_file = fopen(rng_path.c_str(), "r");
  if (rng_file != nullptr) {
    printf("reading rng state from file: %s\n", rng_path.c_str());
    lattice.rng.ReadRng(rng_file);
    fclose(rng_file);
  }

  // check if there is a field checkpoint file
  FILE* field_file = fopen(field_path.c_str(), "rb");
  if (field_file != nullptr) {
    printf("reading field from file: %s\n", field_path.c_str());
    fseek(field_file, 0, SEEK_END);
    assert(ftell(field_file) == field.phi.size() * sizeof(double));
    fseek(field_file, 0, SEEK_SET);
    field.ReadField(field_file);
    fclose(field_file);
  } else if (cold_start) {
    printf("cold start\n");
    field.ColdStart();
  } else {
    printf("hot start\n");
    field.HotStart();
  }
  printf("msq: %.6f\n", field.msq);
  printf("lambda: %.4f\n", field.lambda);
  printf("metropolis_z: %.4f\n", field.metropolis_z);

  double vol = lattice.vol;
  double vol_sq = vol * vol;
  int t_half = n_t / 2 + 1;

  if (use_ricci) {
    // calculate local ricci curvature term
    std::vector<double> local_curvature(lattice.n_distinct);
    for (int id = 0; id < lattice.n_distinct; id++) {
      int s_i = lattice.distinct_first[id] % n_sites_slice;
      Eigen::Vector3d r_ric = Eigen::Vector3d::Zero();
      for (int n = 0; n < lattice.sites[s_i].nn; n++) {
        int s_j = lattice.sites[s_i].neighbors[n] % n_sites_slice;
        int l = lattice.sites[s_i].links[n];
        r_ric += lattice.links[l].wt * (lattice.r[s_i] - lattice.r[s_j]);
      }
      local_curvature[id] = 0.5 * r_ric.norm() / lattice.sites[s_i].wt;
    }

    // apply ricci term to all sites
    for (int s = 0; s < lattice.n_sites; s++) {
      int id = lattice.sites[s].id;
      field.msq_ct[s] = 0.25 * local_curvature[id];  // = a^2 / 4 r^2
    }
  }

  // open the counter term file
  if (!ct_path.empty()) {
    printf("reading counterterm file: %s\n", ct_path.c_str());
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
      field.msq_ct[s] += 96.0 * field.lambda * field.lambda * ct3[id];
    }
  }
  printf("initial action: %.12f\n", field.Action());

  // measurements
  std::vector<std::vector<QfeMeasReal>> legendre_2pt(l_max + 1);
  std::vector<std::vector<QfeMeasReal>> legendre_4pt(l_max + 1);
  for (int l = 0; l <= l_max; l++) {
    legendre_2pt[l].resize(t_half);
    legendre_4pt[l].resize(t_half);
  }
  std::vector<std::vector<QfeMeasReal>> ylm_2pt(n_ylm);
  std::vector<std::vector<QfeMeasReal>> ylm_4pt(n_ylm);
  for (int i_ylm = 0; i_ylm < n_ylm; i_ylm++) {
    ylm_2pt[i_ylm].resize(t_half);
    ylm_4pt[i_ylm].resize(t_half);
  }
  QfeMeasReal anti_2pt;  // antipodal 2-point function
  QfeMeasReal mag;       // magnetization
  QfeMeasReal mag_2;     // magnetization^2
  QfeMeasReal mag_4;     // magnetization^4
  QfeMeasReal mag_6;     // magnetization^6
  QfeMeasReal mag_8;     // magnetization^8
  QfeMeasReal mag_10;    // magnetization^10
  QfeMeasReal mag_12;    // magnetization^12
  QfeMeasReal action;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;
  QfeMeasReal accept_overrelax;
  QfeMeasReal overrelax_demon;

  // read bulk measurements
  FILE* bulk_file = fopen(bulk_path.c_str(), "r");
  if (bulk_file != nullptr) {
    printf("reading measurements from file: %s\n", legendre_2pt_path.c_str());
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
  FILE* legendre_2pt_file = fopen(legendre_2pt_path.c_str(), "r");
  if (legendre_2pt_file != nullptr) {
    printf("reading measurements from file: %s\n", legendre_2pt_path.c_str());
    int l, t;
    while (!feof(legendre_2pt_file)) {
      fscanf(legendre_2pt_file, "%d %d ", &l, &t);
      if (l > l_max || t > t_half) {
        fscanf(legendre_2pt_file, "%*[^\n]");
      } else {
        legendre_2pt[l][t].ReadMeasurement(legendre_2pt_file);
      }
    }
    fclose(legendre_2pt_file);
  }

  // read 4-point function legendre coefficients
  FILE* legendre_4pt_file = fopen(legendre_4pt_path.c_str(), "r");
  if (legendre_4pt_file != nullptr) {
    printf("reading measurements from file: %s\n", legendre_4pt_path.c_str());
    int l, t;
    while (!feof(legendre_4pt_file)) {
      fscanf(legendre_4pt_file, "%d %d ", &l, &t);
      if (l > l_max || t > t_half) {
        fscanf(legendre_4pt_file, "%*[^\n]");
      } else {
        legendre_4pt[l][t].ReadMeasurement(legendre_4pt_file);
      }
    }
    fclose(legendre_4pt_file);
  }

  // read 2-point function ylm coefficients
  FILE* ylm_2pt_file = fopen(ylm_2pt_path.c_str(), "r");
  if (ylm_2pt_file != nullptr) {
    printf("reading measurements from file: %s\n", ylm_2pt_path.c_str());
    int i_ylm, l, m, t;
    while (!feof(ylm_2pt_file)) {
      fscanf(ylm_2pt_file, "%d %d %d %d ", &i_ylm, &l, &m, &t);
      if (i_ylm >= n_ylm || t > t_half) {
        fscanf(ylm_2pt_file, "%*[^\n]");
      } else {
        ylm_2pt[i_ylm][t].ReadMeasurement(ylm_2pt_file);
      }
    }
    fclose(ylm_2pt_file);
  }

  // read 4-point function ylm coefficients
  FILE* ylm_4pt_file = fopen(ylm_4pt_path.c_str(), "r");
  if (ylm_4pt_file != nullptr) {
    printf("reading measurements from file: %s\n", ylm_4pt_path.c_str());
    int i_ylm, l, m, t;
    while (!feof(ylm_4pt_file)) {
      fscanf(ylm_4pt_file, "%d %d %d %d ", &i_ylm, &l, &m, &t);
      if (i_ylm >= n_ylm || t > t_half) {
        fscanf(ylm_4pt_file, "%*[^\n]");
      } else {
        ylm_4pt[i_ylm][t].ReadMeasurement(ylm_4pt_file);
      }
    }
    fclose(ylm_4pt_file);
  }

  Timer timer;

  for (int n = 0; n < (n_traj + n_therm); n++) {
    // terminate if wall time is reached
    if (wall_time > 0.0 && timer.Duration() > wall_time) break;

    // do wolff updates
    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      cluster_size_sum += field.WolffUpdate();
    }
    cluster_size.Measure(double(cluster_size_sum) / vol);

    // do metropolis updates
    double metropolis_sum = 0.0;
    for (int j = 0; j < n_metropolis; j++) {
      metropolis_sum += field.Metropolis();
    }
    accept_metropolis.Measure(metropolis_sum);

    // do overrelaxation update
    if (do_overrelax) {
      accept_overrelax.Measure(field.Overrelax());
    }

    // continue if not measuring this configuration
    if (n % n_skip || n < n_therm) continue;

    // measure overrelaxation demon
    if (do_overrelax) {
      overrelax_demon.Measure(field.overrelax_demon);
    }

    // measure correlators
    std::vector<std::vector<Complex>> ylm_2pt_sum(n_ylm);
    std::vector<std::vector<Complex>> ylm_4pt_sum(n_ylm);
    for (int i_ylm = 0; i_ylm < n_ylm; i_ylm++) {
      ylm_2pt_sum[i_ylm].resize(n_t, 0.0);
      ylm_4pt_sum[i_ylm].resize(n_t, 0.0);
    }

    double mag_sum = 0.0;
    double anti_2pt_sum = 0.0;

    // measure phi_lm by summing over all sites
    for (int s = 0; s < lattice.n_sites; s++) {
      int s0 = s % n_sites_slice;
      int t = s / n_sites_slice;
      int a = lattice.antipode[s0] + t * n_sites_slice;
      double wt_2pt = field.phi[s] * lattice.sites[s].wt;
      double wt_4pt = wt_2pt * field.phi[a];

      mag_sum += wt_2pt;
      anti_2pt_sum += wt_4pt;

      for (int ylm_i = 0; ylm_i < n_ylm; ylm_i++) {
        Complex y = lattice.ylm[s0][ylm_i];
        ylm_2pt_sum[ylm_i][t] += y * wt_2pt;
        ylm_4pt_sum[ylm_i][t] += y * wt_4pt;
      }
    }

    // measure Ylm coefficients of the 2-point and 4-point function, then
    // use the addition theorem to calculate the Legendre coefficients
    std::vector<double> ylm_2pt_partial(t_half, 0.0);
    std::vector<double> ylm_4pt_partial(t_half, 0.0);
    std::vector<double> legendre_2pt_sum(t_half, 0.0);
    std::vector<double> legendre_4pt_sum(t_half, 0.0);

    for (int ylm_i = 0, l = 0, m = 0; ylm_i < n_ylm; ylm_i++) {
      // sum over all pairs of time slices
      for (int t1 = 0; t1 < n_t; t1++) {
        for (int t2 = t1; t2 < n_t; t2++) {
          // this calculates the shortest distance between t1 and t2
          int dt = (n_t - abs(2 * abs(t1 - t2) - n_t)) / 2;

          // double counting factor for t == n_t / 2
          double t_mult = 1.0;
          if (dt == (t_half - 1)) t_mult *= 2.0;

          Complex y2 = ylm_2pt_sum[ylm_i][t1] * conj(ylm_2pt_sum[ylm_i][t2]);
          Complex y4 = ylm_4pt_sum[ylm_i][t1] * conj(ylm_4pt_sum[ylm_i][t2]);
          ylm_2pt_partial[dt] += real(y2) * t_mult;
          ylm_4pt_partial[dt] += real(y4) * t_mult;
        }
      }

      // coefficient for addition theorem
      // includes double counting factor to include negative m values
      double lm_mult = 4.0 * M_PI / double(2 * l + 1) * (m == 0 ? 1.0 : 2.0);

      for (int dt = 0; dt < t_half; dt++) {
        double ylm_2pt_value = ylm_2pt_partial[dt] / vol_sq * double(n_t);
        double ylm_4pt_value = ylm_4pt_partial[dt] / vol_sq * double(n_t);
        ylm_2pt[ylm_i][dt].Measure(ylm_2pt_value, false);
        ylm_4pt[ylm_i][dt].Measure(ylm_4pt_value, false);
        ylm_2pt_partial[dt] = 0.0;
        ylm_4pt_partial[dt] = 0.0;
        legendre_2pt_sum[dt] += lm_mult * ylm_2pt_value;
        legendre_4pt_sum[dt] += lm_mult * ylm_4pt_value;
      }

      // go to next m until m == l
      m++;
      if (m <= l) continue;

      // measure Legendre coefficients
      for (int dt = 0; dt < t_half; dt++) {
        legendre_2pt[l][dt].Measure(legendre_2pt_sum[dt], false);
        legendre_4pt[l][dt].Measure(legendre_4pt_sum[dt], false);
        legendre_2pt_sum[dt] = 0.0;
        legendre_4pt_sum[dt] = 0.0;
      }
      l++;
      m = 0;
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
    anti_2pt.Measure(anti_2pt_sum / lattice.vol);
    action.Measure(field.Action());
    if (verbose) {
      printf("%06d %.12f %.4f %.4f\n", n, action.last, accept_metropolis.last,
             cluster_size.last);
    }
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

  // open the bulk data file
  printf("writing measurements to file: %s\n", bulk_path.c_str());
  bulk_file = fopen(bulk_path.c_str(), "w");
  assert(bulk_file != nullptr);

  printf("action: %+.12e %.12e %.4f %.4f\n", action.Mean(), action.Error(),
         action.AutocorrFront(), action.AutocorrBack());
  fprintf(bulk_file, "action %.16e %.16e %d\n", action.Mean(), action.Error(),
          action.n);
  printf("mag: %.12e %.12e %.4f %.4f\n", m_mean, m_err, mag.AutocorrFront(),
         mag.AutocorrBack());
  fprintf(bulk_file, "mag %.16e %.16e %d\n", m_mean, m_err, mag.n);
  printf("m^2: %.12e %.12e %.4f %.4f\n", m2_mean, m2_err, mag_2.AutocorrFront(),
         mag_2.AutocorrBack());
  fprintf(bulk_file, "mag^2 %.16e %.16e %d\n", m2_mean, m2_err, mag_2.n);
  printf("m^4: %.12e %.12e %.4f %.4f\n", m4_mean, m4_err, mag_4.AutocorrFront(),
         mag_4.AutocorrBack());
  fprintf(bulk_file, "mag^4 %.16e %.16e %d\n", m4_mean, m4_err, mag_4.n);
  printf("m^6: %.12e %.12e %.4f %.4f\n", mag_6.Mean(), mag_6.Error(),
         mag_6.AutocorrFront(), mag_6.AutocorrBack());
  fprintf(bulk_file, "mag^6 %.16e %.16e %d\n", mag_6.Mean(), mag_6.Error(),
          mag_6.n);
  printf("m^8: %.12e %.12e %.4f %.4f\n", mag_8.Mean(), mag_8.Error(),
         mag_8.AutocorrFront(), mag_8.AutocorrBack());
  fprintf(bulk_file, "mag^8 %.16e %.16e %d\n", mag_8.Mean(), mag_8.Error(),
          mag_8.n);
  printf("m^10: %.12e %.12e %.4f %.4f\n", mag_10.Mean(), mag_10.Error(),
         mag_10.AutocorrFront(), mag_10.AutocorrBack());
  fprintf(bulk_file, "mag^10 %.16e %.16e %d\n", mag_10.Mean(), mag_10.Error(),
          mag_10.n);
  printf("m^12: %.12e %.12e %.4f %.4f\n", mag_12.Mean(), mag_12.Error(),
         mag_12.AutocorrFront(), mag_12.AutocorrBack());
  fprintf(bulk_file, "mag^12 %.16e %.16e %d\n", mag_12.Mean(), mag_12.Error(),
          mag_12.n);
  printf("anti_2pt: %.12e %.12e %.4f %.4f\n", anti_2pt.Mean(), anti_2pt.Error(),
         anti_2pt.AutocorrFront(), anti_2pt.AutocorrBack());
  fprintf(bulk_file, "anti_2pt %.16e %.16e %d\n", anti_2pt.Mean(),
          anti_2pt.Error(), anti_2pt.n);
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
  printf("writing measurements to file: %s\n", legendre_2pt_path.c_str());
  legendre_2pt_file = fopen(legendre_2pt_path.c_str(), "w");
  assert(legendre_2pt_file != nullptr);
  for (int l = 0; l <= l_max; l++) {
    for (int t = 0; t < t_half; t++) {
      fprintf(legendre_2pt_file, "%02d %04d ", l, t);
      legendre_2pt[l][t].WriteMeasurement(legendre_2pt_file);
    }
  }
  fclose(legendre_2pt_file);

  // print 4-point function legendre coefficients
  printf("writing measurements to file: %s\n", legendre_4pt_path.c_str());
  legendre_4pt_file = fopen(legendre_4pt_path.c_str(), "w");
  assert(legendre_4pt_file != nullptr);
  for (int l = 0; l <= l_max; l += 2) {
    for (int t = 0; t < t_half; t++) {
      fprintf(legendre_4pt_file, "%02d %04d ", l, t);
      legendre_4pt[l][t].WriteMeasurement(legendre_4pt_file);
    }
  }
  fclose(legendre_4pt_file);

  // print 2-point function spherical harmonic coefficients
  printf("writing measurements to file: %s\n", ylm_2pt_path.c_str());
  ylm_2pt_file = fopen(ylm_2pt_path.c_str(), "w");
  assert(ylm_2pt_file != nullptr);
  for (int ylm_i = 0, l = 0, m = 0; ylm_i < n_ylm; ylm_i++) {
    for (int t = 0; t < t_half; t++) {
      fprintf(ylm_2pt_file, "%04d %02d %02d %04d ", ylm_i, l, m, t);
      ylm_2pt[ylm_i][t].WriteMeasurement(ylm_2pt_file);
    }
    m++;
    if (m > l) {
      l++;
      m = 0;
    }
  }
  fclose(ylm_2pt_file);

  // print 4-point function spherical harmonic coefficients
  printf("writing measurements to file: %s\n", ylm_4pt_path.c_str());
  ylm_4pt_file = fopen(ylm_4pt_path.c_str(), "w");
  assert(ylm_4pt_file != nullptr);
  for (int ylm_i = 0, l = 0, m = 0; ylm_i < n_ylm; ylm_i++) {
    if ((l % 2) == 0) {
      for (int t = 0; t < t_half; t++) {
        fprintf(ylm_4pt_file, "%04d %02d %02d %04d ", ylm_i, l, m, t);
        ylm_4pt[ylm_i][t].WriteMeasurement(ylm_4pt_file);
      }
    }
    m++;
    if (m > l) {
      l++;
      m = 0;
    }
  }
  fclose(ylm_4pt_file);

  return 0;
}
