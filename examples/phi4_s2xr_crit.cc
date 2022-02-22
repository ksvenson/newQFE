// phi4_s2xr_crit.cc

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

#include <boost/math/special_functions/legendre.hpp>
using boost::math::legendre_p;

typedef std::complex<double> Complex;

int main(int argc, char* argv[]) {

  // default parameters
  int n_refine = 4;
  int n_t = 64;
  int q = 5;
  double msq = -0.2702 * 2.0;
  double lambda = 0.2;
  const char* ct_dir = "./ct";
  unsigned int seed = 1234u;
  bool cold_start = false;
  int l_max = 6;
  int n_therm = 2000;
  int n_traj = 20000;
  int n_skip = 20;
  int n_wolff = 5;
  int n_metropolis = 4;
  double metropolis_z = 1.0;
  std::string data_dir = "phi4_s2xr_crit";

  const struct option long_options[] = {
    { "n_refine", required_argument, 0, 'N' },
    { "n_t", required_argument, 0, 'T' },
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

  const char* short_options = "N:T:q:S:Cm:L:c:l:h:t:s:w:e:z:d:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N': n_refine = atoi(optarg); break;
      case 'T': n_t = atoi(optarg); break;
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
  int n_sites_slice = lattice.n_sites;
  lattice.AddDimension(n_t);
  for (int s = n_sites_slice; s < lattice.n_sites; s++) {
    int s0 = s % n_sites_slice;
    int t = s / n_sites_slice;
    lattice.r[s] = lattice.r[s0];
    lattice.antipode[s] = lattice.antipode[s0] + t * n_sites_slice;
    lattice.ylm[s].resize(n_ylm);
    for (int i_ylm = 0; i_ylm < n_ylm; i_ylm++) {
      lattice.ylm[s][i_ylm] = lattice.ylm[s0][i_ylm];
    }
  }
  printf("n_refine: %d\n", n_refine);
  printf("n_t: %d\n", n_t);
  printf("q: %d\n", q);
  printf("total sites: %d\n", lattice.n_sites);

  // add ricci term
  double msq_ricci = M_PI / double(n_sites_slice);
  QfePhi4 field(&lattice, msq + msq_ricci, lambda);
  if (cold_start) {
    printf("cold start\n");
    field.ColdStart();
  } else {
    printf("hot start\n");
    field.HotStart();
  }
  field.metropolis_z = metropolis_z;
  printf("msq: %.4f\n", msq);
  printf("msq: %.4f (w/ ricci term)\n", field.msq);
  printf("lambda: %.4f\n", field.lambda);
  printf("metropolis_z: %.4f\n", field.metropolis_z);
  printf("initial action: %.12f\n", field.Action());

  double vol = lattice.vol;
  double vol_sq = vol * vol;
  int t_half = n_t / 2 + 1;

  // open the counter term file
  char path[200];
  sprintf(path, "%s/ct_%d_%d_%d.dat", ct_dir, q, n_refine, n_t);
  FILE* ct_file = fopen(path, "r");
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
    field.msq_ct[s] += -96.0 * field.lambda * field.lambda * ct3[id];
  }

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
  QfeMeasReal mag;  // magnetization
  QfeMeasReal mag_2;  // magnetization^2
  QfeMeasReal mag_4;  // magnetization^4
  QfeMeasReal action;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;

  Timer timer;

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
    field.SWUpdate();

#if 0
    // **** cut ****

    std::vector<std::vector<double>> legendre_2pt_sum(l_max + 1);
    std::vector<std::vector<double>> legendre_4pt_sum(l_max + 1);
    std::vector<std::vector<double>> ylm_2pt_sum(n_ylm);
    std::vector<std::vector<double>> ylm_4pt_sum(n_ylm);

    for (int l = 0; l <= l_max; l++) {
      legendre_2pt_sum[l].resize(t_half, 0.0);
      legendre_4pt_sum[l].resize(t_half, 0.0);
    }
    for (int i_ylm = 0; i_ylm < n_ylm; i_ylm++) {
      ylm_2pt_sum[i_ylm].resize(t_half, 0.0);
      ylm_4pt_sum[i_ylm].resize(t_half, 0.0);
    }
    double mag_sum = 0.0;
    double anti_2pt_sum = 0.0;

    for (int s1 = 0; s1 < lattice.n_sites; s1++) {
      int a1 = lattice.antipode[s1];
      int t1 = s1 / n_sites_slice;
      double phi_1 = field.phi[s1] * lattice.sites[s1].wt;
      // double phi_11 = phi_1 * phi_1;
      int r1 = field.sw_root[s1];
      int r3 = field.sw_root[a1];

      mag_sum += phi_1;
      anti_2pt_sum += phi_1 * field.phi[a1];

      // for (int l = 0; l <= l_max; l++) {
      //   double legendre_val = legendre_p(l, 1.0) * phi_11;
      //   legendre_2pt_sum[l][0] += legendre_val;
      //   legendre_4pt_sum[l][0] += legendre_val * field.phi[a1] * field.phi[a1];
      // }
      //
      // for (int ylm_i = 0; ylm_i < n_ylm; ylm_i++) {
      //   Complex y = lattice.ylm[s1][ylm_i];
      //   double ylm_val = std::norm(y) * phi_11;
      //   ylm_2pt_sum[ylm_i][0] += ylm_val;
      //   ylm_4pt_sum[ylm_i][0] += ylm_val * field.phi[a1] * field.phi[a1];
      // }

      for (int s2 = 0; s2 < lattice.n_sites; s2++) {
        int a2 = lattice.antipode[s2];
        int t2 = s2 / n_sites_slice;
        if (t2 > t1) continue;
        double phi_2pt = phi_1 * field.phi[s2] * lattice.sites[s2].wt;
        double phi_4pt = phi_2pt * field.phi[a1] * field.phi[a2];
        int r2 = field.sw_root[s2];
        int r4 = field.sw_root[a2];

        if (r1 == r2) {
          // 2-point function is nonzero

          if (r3 != r4) {
            // antipodal points are not in the same cluster
            phi_4pt = 0.0;
          }
        } else if ((r1 == r3 && r2 == r4) || (r1 == r4 && r2 == r3)) {
          // 4-point function is nonzero
          phi_2pt = 0.0;
        } else {
          continue;
        }

        int dt = (n_t - abs(2 * abs(t1 - t2) - n_t)) / 2;

        if (dt == (t_half - 1)) {
          phi_2pt *= 2.0;
          phi_4pt *= 2.0;
        }

        double cos_theta = lattice.CosTheta(s1, s2);
        if (cos_theta > 1.0) cos_theta = 1.0;
        if (cos_theta < -1.0) cos_theta = -1.0;
        for (int l = 0; l <= l_max; l++) {
          double legendre_val = legendre_p(l, cos_theta);
          legendre_2pt_sum[l][dt] += legendre_val * phi_2pt;
          legendre_4pt_sum[l][dt] += legendre_val * phi_4pt;
        }

        for (int ylm_i = 0; ylm_i < n_ylm; ylm_i++) {
          Complex y1 = lattice.ylm[s1][ylm_i];
          Complex y2 = lattice.ylm[s2][ylm_i];
          double ylm_val = real(y1 * conj(y2));
          ylm_2pt_sum[ylm_i][dt] += ylm_val * phi_2pt;
          ylm_4pt_sum[ylm_i][dt] += ylm_val * phi_4pt;
        }
      }
    }

    for (int dt = 0; dt < t_half; dt++) {
      for (int l = 0; l <= l_max; l++) {
        legendre_2pt[l][dt].Measure(legendre_2pt_sum[l][dt] / vol_sq * double(n_t));
        legendre_4pt[l][dt].Measure(legendre_4pt_sum[l][dt] / vol_sq * double(n_t));
      }
      for (int ylm_i = 0; ylm_i < n_ylm; ylm_i++) {
        ylm_2pt[ylm_i][dt].Measure(ylm_2pt_sum[ylm_i][dt] / vol_sq);
        ylm_4pt[ylm_i][dt].Measure(ylm_4pt_sum[ylm_i][dt] / vol_sq);
      }
    }

    // **** cut ****
#else
    // measure correlators
    std::vector<std::vector<Complex>> ylm_2pt_sum(n_ylm);
    std::vector<std::vector<Complex>> ylm_4pt_sum(n_ylm);
    for (int i_ylm = 0; i_ylm < n_ylm; i_ylm++) {
      ylm_2pt_sum[i_ylm].resize(n_t, 0.0);
      ylm_4pt_sum[i_ylm].resize(n_t, 0.0);
    }

    double mag_sum = 0.0;
    double anti_2pt_sum = 0.0;

    for (int s = 0; s < lattice.n_sites; s++) {
      int t = s / n_sites_slice;
      int a = lattice.antipode[s];
      double wt_2pt = field.phi[s] * lattice.sites[s].wt;
      double wt_4pt = wt_2pt * field.phi[a];

      mag_sum += wt_2pt;
      anti_2pt_sum += wt_4pt;

      for (int ylm_i = 0; ylm_i < n_ylm; ylm_i++) {
        Complex y = lattice.ylm[s][ylm_i];
        ylm_2pt_sum[ylm_i][t] += y * wt_2pt;
        ylm_4pt_sum[ylm_i][t] += y * wt_4pt;
      }
    }

    std::vector<double> legendre_2pt_sum(t_half, 0.0);
    std::vector<double> legendre_4pt_sum(t_half, 0.0);
    for (int ylm_i = 0, l = 0, m = 0; ylm_i < n_ylm; ylm_i++) {
      // sum over pairs of time slices
      for (int t1 = 0; t1 < n_t; t1++) {
        for (int t2 = t1; t2 < n_t; t2++) {
          int dt = (n_t - abs(2 * abs(t1 - t2) - n_t)) / 2;
          Complex y2 = ylm_2pt_sum[ylm_i][t1] * conj(ylm_2pt_sum[ylm_i][t2]);
          Complex y4 = ylm_4pt_sum[ylm_i][t1] * conj(ylm_4pt_sum[ylm_i][t2]);
          ylm_2pt[ylm_i][dt].Measure(real(y2) / vol_sq * double(n_t));
          ylm_4pt[ylm_i][dt].Measure(real(y4) / vol_sq * double(n_t));

          legendre_2pt_sum[dt] += ylm_2pt[ylm_i][dt].last * (m == 0 ? 1.0 : 2.0);
          legendre_4pt_sum[dt] += ylm_4pt[ylm_i][dt].last * (m == 0 ? 1.0 : 2.0);
        }
      }
      m++;
      if (m > l) {
        for (int dt = 0; dt < t_half; dt++) {
          double coeff = 4.0 * M_PI / double(2 * l + 1);
          if (dt == (t_half - 1)) coeff *= 2.0;
          legendre_2pt[l][dt].Measure(legendre_2pt_sum[dt] * coeff);
          legendre_4pt[l][dt].Measure(legendre_4pt_sum[dt] * coeff);
          legendre_2pt_sum[dt] = 0.0;
          legendre_4pt_sum[dt] = 0.0;
        }
        l++;
        m = 0;
      }
    }
  #endif

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

    if (timer.Duration() > 3600.0) break;
  }

  timer.Stop();
  printf("duration: %.6f\n", timer.Duration());

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
  sprintf(run_id, "%d_%d_%d", q, n_refine, n_t);
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
    for (int t = 0; t < t_half; t++) {
      printf("legendre_2pt_%02d(%04d): %.12e", l, t, legendre_2pt[l][t].Mean());
      printf(" %.12e", legendre_2pt[l][t].Error());
      printf(" %.4f", legendre_2pt[l][t].AutocorrFront());
      printf(" %.4f\n", legendre_2pt[l][t].AutocorrBack());
      fprintf(file, "%04d %04d %.16e %.16e %d\n", l, t, \
          legendre_2pt[l][t].Mean(), legendre_2pt[l][t].Error(), \
          legendre_2pt[l][t].n);
    }
  }
  fclose(file);

  // print 4-point function legendre coefficients
  sprintf(path, "%s/%s/%s_legendre_4pt_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", path);
  file = fopen(path, "w");
  assert(file != nullptr);
  for (int l = 0; l <= l_max; l++) {
    for (int t = 0; t < t_half; t++) {
      printf("legendre_4pt_%02d(%04d): %.12e", l, t, legendre_4pt[l][t].Mean());
      printf(" %.12e", legendre_4pt[l][t].Error());
      printf(" %.4f", legendre_4pt[l][t].AutocorrFront());
      printf(" %.4f\n", legendre_4pt[l][t].AutocorrBack());
      fprintf(file, "%04d %04d %.16e %.16e %d\n", l, t, \
          legendre_4pt[l][t].Mean(), legendre_4pt[l][t].Error(), \
          legendre_4pt[l][t].n);
    }
  }
  fclose(file);

  // print 2-point function spherical harmonic coefficients
  sprintf(path, "%s/%s/%s_ylm_2pt_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", path);
  file = fopen(path, "w");
  assert(file != nullptr);
  for (int ylm_i = 0, l = 0, m = 0; ylm_i < n_ylm; ylm_i++) {
    for (int t = 0; t < t_half; t++) {
      printf("ylm_2pt_%02d_%02d(%04d): %.12e", l, m, t, ylm_2pt[ylm_i][t].Mean());
      printf(" %.12e", ylm_2pt[ylm_i][t].Error());
      printf(" %.4f", ylm_2pt[ylm_i][t].AutocorrFront());
      printf(" %.4f\n", ylm_2pt[ylm_i][t].AutocorrBack());
      fprintf(file, "%04d %04d %.16e %.16e %d\n", ylm_i, t, \
          ylm_2pt[ylm_i][t].Mean(), ylm_2pt[ylm_i][t].Error(), \
          ylm_2pt[ylm_i][t].n);
    }
    m++;
    if (m > l) {
      l++;
      m = 0;
    }
  }
  fclose(file);

  // print 4-point function spherical harmonic coefficients
  sprintf(path, "%s/%s/%s_ylm_4pt_%08X.dat", \
      data_dir.c_str(), run_id, run_id, seed);
  printf("opening file: %s\n", path);
  file = fopen(path, "w");
  assert(file != nullptr);
  for (int ylm_i = 0, l = 0, m = 0; ylm_i < n_ylm; ylm_i++) {
    for (int t = 0; t < t_half; t++) {
      printf("ylm_4pt_%02d_%02d(%04d): %.12e", l, m, t, ylm_4pt[ylm_i][t].Mean());
      printf(" %.12e", ylm_4pt[ylm_i][t].Error());
      printf(" %.4f", ylm_4pt[ylm_i][t].AutocorrFront());
      printf(" %.4f\n", ylm_4pt[ylm_i][t].AutocorrBack());
      fprintf(file, "%04d %04d %.16e %.16e %d\n", ylm_i, t, \
          ylm_4pt[ylm_i][t].Mean(), ylm_4pt[ylm_i][t].Error(), \
          ylm_4pt[ylm_i][t].n);
    }
    m++;
    if (m > l) {
      l++;
      m = 0;
    }
  }
  fclose(file);

  return 0;
}
