// ising_s2_crit.cc

#include <getopt.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <boost/math/special_functions/legendre.hpp>
#include "ising.h"
#include "s2.h"
#include "statistics.h"

typedef std::complex<double> Complex;
using boost::math::legendre_p;

int main(int argc, char* argv[]) {

  // default parameters
  int n_refine = 8;
  int q = 5;
  unsigned int seed = 1234u;
  int l_max = 6;
  int n_therm = 2000;
  int n_traj = 100000;
  int n_skip = 10;
  int n_wolff = 5;
  int n_metropolis = 4;
  std::string data_dir = "ising_s2_crit";

  const struct option long_options[] = {
    { "n_refine", required_argument, 0, 'N' },
    { "q", required_argument, 0, 'q' },
    { "seed", required_argument, 0, 'S' },
    { "l_max", required_argument, 0, 'l' },
    { "n_therm", required_argument, 0, 'h' },
    { "n_traj", required_argument, 0, 't' },
    { "n_skip", required_argument, 0, 's' },
    { "n_wolff", required_argument, 0, 'w' },
    { "n_metropolis", required_argument, 0, 'e' },
    { "data_dir", required_argument, 0, 'd' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "N:q:S:l:h:t:s:w:e:d:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N': n_refine = atoi(optarg); break;
      case 'q': q = atoi(optarg); break;
      case 'S': seed = atol(optarg); break;
      case 'l': l_max = atoi(optarg); break;
      case 'h': n_therm = atoi(optarg); break;
      case 't': n_traj = atoi(optarg); break;
      case 's': n_skip = atoi(optarg); break;
      case 'w': n_wolff = atoi(optarg); break;
      case 'e': n_metropolis = atoi(optarg); break;
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

  QfeIsing field(&lattice, 1.0);
  field.HotStart();
  printf("initial action: %.12f\n", field.Action());

  double vol = lattice.vol;
  double vol_sq = vol * vol;

  for (int l = 0; l < lattice.n_links; l++) {

    double sq_edge = lattice.EdgeSquared(l);
    double edge_dual = 0.0;

    for (int i = 0; i < 2; i++) {

      // find the other two edges of this face
      int f = lattice.links[l].faces[i];
      int e = 0;
      if (lattice.faces[f].edges[0] == l) {
        e = 0;
      } else if (lattice.faces[f].edges[1] == l) {
        e = 1;
      } else if (lattice.faces[f].edges[2] == l) {
        e = 2;
      } else {
        printf("invalid face %04d for link %04d\n", f, l);
      }
      int e1 = (e + 1) % 3;
      int e2 = (e + 2) % 3;
      int l1 = lattice.faces[f].edges[e1];
      int l2 = lattice.faces[f].edges[e2];

      // find the area associated with this face
      double sq_edge_1 = lattice.EdgeSquared(l1);
      double sq_edge_2 = lattice.EdgeSquared(l2);
      edge_dual += 0.125 * (sq_edge_1 + sq_edge_2 - sq_edge) / lattice.FlatArea(f);
    }
    lattice.links[l].wt = 0.5 * asinh(edge_dual);
  }

/** numbering for adjacent sites, links, and faces
 *
 *                       s2
 *                     /    \
 *                   /        \
 *                l1            l2
 *               /       f0       \
 *             /                    \
 *           s0 - - - -  l0  - - - - s1
 *             \                    /
 *               \       f1       /
 *                l3            l4
 *                   \        /
 *                     \    /
 *                       s3
 *
 */

  for (int l0 = 0; l0 < lattice.n_links; l0++) {

    int f0 = lattice.links[l0].faces[0];
    int f1 = lattice.links[l0].faces[1];

    int s0 = lattice.links[l0].sites[0];
    int s1 = lattice.links[l0].sites[1];
    int s2, s3;

    int l1, l2, l3, l4;

    for (int i = 0; i < 3; i++) {
      int e = lattice.faces[f0].edges[i];
      if (e == l0) continue;
      int s_a = lattice.links[e].sites[0];
      int s_b = lattice.links[e].sites[1];
      if (s_a != s0 && s_a != s1) {
        s2 = s_a;
        if (s_b == s0) {
          l1 = e;
        } else {
          l2 = e;
        }
      } else if (s_b != s0 && s_b != s1) {
        s2 = s_b;
        if (s_a == s0) {
          l1 = e;
        } else {
          l2 = e;
        }
      } else {
        printf("invalid edge %04d for face %04d\n", e, f0);
      }
    }

    for (int i = 0; i < 3; i++) {
      int e = lattice.faces[f1].edges[i];
      if (e == l0) continue;
      int s_a = lattice.links[e].sites[0];
      int s_b = lattice.links[e].sites[1];
      if (s_a != s0 && s_a != s1) {
        s3 = s_a;
        if (s_b == s0) {
          l3 = e;
        } else {
          l4 = e;
        }
      } else if (s_b != s0 && s_b != s1) {
        s3 = s_b;
        if (s_a == s0) {
          l3 = e;
        } else {
          l4 = e;
        }
      } else {
        printf("invalid edge %04d for face %04d\n", e, f0);
      }
    }

    // calculate unit vectors from links to/from face circumcenters
    Eigen::Vector3d v_f0 = lattice.FaceCircumcenter(f0);
    Eigen::Vector3d v_f1 = lattice.FaceCircumcenter(f1);
    Eigen::Vector3d v_l0 = 0.5 * (lattice.r[s0] + lattice.r[s1]);
    Eigen::Vector3d v_l1 = 0.5 * (lattice.r[s0] + lattice.r[s2]);
    Eigen::Vector3d v_l2 = 0.5 * (lattice.r[s1] + lattice.r[s2]);
    Eigen::Vector3d v_l3 = 0.5 * (lattice.r[s0] + lattice.r[s3]);
    Eigen::Vector3d v_l4 = 0.5 * (lattice.r[s1] + lattice.r[s3]);
    Eigen::Vector3d v_l0_f0 = (v_f0 - v_l0).normalized();
    Eigen::Vector3d v_l0_f1 = (v_f1 - v_l0).normalized();
    Eigen::Vector3d v_f0_l1 = (v_l1 - v_f0).normalized();
    Eigen::Vector3d v_f0_l2 = (v_l2 - v_f0).normalized();
    Eigen::Vector3d v_f1_l3 = (v_l3 - v_f1).normalized();
    Eigen::Vector3d v_f1_l4 = (v_l4 - v_f1).normalized();
    double cos1 = sqrt(0.5 * (1.0 + v_l0_f0.dot(v_f0_l1)));
    double cos2 = sqrt(0.5 * (1.0 + v_l0_f0.dot(v_f0_l2)));
    double cos3 = sqrt(0.5 * (1.0 + v_l0_f1.dot(v_f1_l3)));
    double cos4 = sqrt(0.5 * (1.0 + v_l0_f1.dot(v_f1_l4)));
    double cos12 = sqrt(0.5 * (1.0 - v_f0_l1.dot(v_f0_l2)));
    double cos34 = sqrt(0.5 * (1.0 - v_f1_l3.dot(v_f1_l4)));
    double cos_prod_num = cos1 * cos2 * cos3 * cos4;
    double cos_prod_den = cos12 * cos34;

    double len_l0_sq = lattice.EdgeSquared(l0);
    double len_l0 = sqrt(len_l0_sq);
    double len_l1 = lattice.EdgeLength(l1);
    double len_l2 = lattice.EdgeLength(l2);
    double len_l3 = lattice.EdgeLength(l3);
    double len_l4 = lattice.EdgeLength(l4);
    double len_f0 = len_l0 + len_l1 + len_l2;
    double len_f1 = len_l0 + len_l3 + len_l4;

    double tanh_sq_L_num = 4.0 * len_l0_sq * cos_prod_num;
    double tanh_sq_L_den = len_f0 * len_f1 * cos_prod_den;
    double L = atanh(sqrt(tanh_sq_L_num / tanh_sq_L_den));
    double K = 0.5 * asinh(1.0 / sinh(2.0 * L));

    // printf("%.12f %.12f\n", lattice.links[l0].wt, K);
    lattice.links[l0].wt = K;
  }

  // measurements
  std::vector<QfeMeasReal> ylm_2pt(n_ylm);
  std::vector<QfeMeasReal> ylm_4pt(n_ylm);
  std::vector<QfeMeasReal> legendre_2pt(l_max + 1);
  std::vector<QfeMeasReal> legendre_4pt(l_max + 1);
  std::vector<double> ylm_2pt_sum(n_ylm);
  std::vector<double> ylm_4pt_sum(n_ylm);
  std::vector<double> legendre_2pt_sum(l_max + 1);
  std::vector<double> legendre_4pt_sum(l_max + 1);
  QfeMeasReal spin;  // average spin (magnetization)
  QfeMeasReal mag_2;  // magnetization^2
  QfeMeasReal mag_4;  // magnetization^4
  QfeMeasReal action;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;
  QfeMeasReal accept_overrelax;

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
    int n_clusters = field.SWUpdate();

    // measure correlators
    std::vector<double>::iterator it;
    it = legendre_2pt_sum.begin();
    std::fill(it, it + l_max + 1, 0.0);
    it = legendre_4pt_sum.begin();
    std::fill(it, it + l_max + 1, 0.0);
    it = ylm_2pt_sum.begin();
    std::fill(it, it + n_ylm, 0.0);
    it = ylm_4pt_sum.begin();
    std::fill(it, it + n_ylm, 0.0);

    for (int c = 0; c < n_clusters; c++) {
      int count = field.sw_clusters[c].size();
      for (int i1 = 0; i1 < count; i1++) {
        int s1 = field.sw_clusters[c][i1];
        int a1 = lattice.antipode[s1];
        double wt_1 = lattice.sites[s1].wt;
        double wt_11 = wt_1 * wt_1;

        for (int l = 0; l <= l_max; l++) {
          double legendre_val = legendre_p(l, 1.0) * wt_11;
          legendre_2pt_sum[l] += legendre_val;
          legendre_4pt_sum[l] += legendre_val;
        }

        for (int ylm_i = 0; ylm_i < n_ylm; ylm_i++) {
          Complex y = lattice.ylm[s1][ylm_i];
          double ylm_val = std::norm(y) * wt_11;
          ylm_2pt_sum[ylm_i] += ylm_val;
          ylm_4pt_sum[ylm_i] += ylm_val;
        }

        for (int i2 = i1 + 1; i2 < count; i2++) {
          int s2 = field.sw_clusters[c][i2];
          int a2 = lattice.antipode[s2];
          double wt_12 = wt_1 * lattice.sites[s2].wt;
          double wt_34 = 0.0;
          if (field.sw_root[a1] == field.sw_root[a2]) {
            // antipodal points are in the same cluster
            wt_34 = wt_12;
          }

          double cos_theta = lattice.CosTheta(s1, s2);
          for (int l = 0; l <= l_max; l++) {
            double legendre_val = 2.0 * legendre_p(l, cos_theta);
            legendre_2pt_sum[l] += legendre_val * wt_12;
            legendre_4pt_sum[l] += legendre_val * wt_34;
          }

          for (int ylm_i = 0; ylm_i < n_ylm; ylm_i++) {
            Complex y1 = lattice.ylm[s1][ylm_i];
            Complex y2 = lattice.ylm[s2][ylm_i];
            double ylm_val = 2.0 * real(y1 * conj(y2));
            ylm_2pt_sum[ylm_i] += ylm_val * wt_12;
            ylm_4pt_sum[ylm_i] += ylm_val * wt_34;
          }
        }
      }
    }

    for (int l = 0; l <= l_max; l++) {
      legendre_2pt[l].Measure(legendre_2pt_sum[l] / vol_sq);
      legendre_4pt[l].Measure(legendre_4pt_sum[l] / vol_sq);
    }
    for (int ylm_i = 0; ylm_i < n_ylm; ylm_i++) {
      ylm_2pt[ylm_i].Measure(ylm_2pt_sum[ylm_i] / vol_sq);
      ylm_4pt[ylm_i].Measure(ylm_4pt_sum[ylm_i] / vol_sq);
    }

    // measure magnetization
    double spin_sum = 0.0;
    for (int s = 0; s < lattice.n_sites; s++) {
      spin_sum += field.spin[s] * lattice.sites[s].wt;
    }
    double m_sq = spin_sum * spin_sum;
    spin.Measure(fabs(spin_sum));
    mag_2.Measure(m_sq);
    mag_4.Measure(m_sq * m_sq);
    action.Measure(field.Action());
    printf("%06d %.12f %.4f %.4f %04d\n", \
        n, action.last, \
        accept_metropolis.last, \
        cluster_size.last, \
        n_clusters);
  }

  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());

  double m_mean = spin.Mean();
  double m_err = spin.Error();
  double m2_mean = mag_2.Mean();
  double m2_err = mag_2.Error();
  double m4_mean = mag_4.Mean();
  double m4_err = mag_4.Error();

  printf("action: %+.12e %.12e %.4f %.4f\n", \
      action.Mean(), action.Error(), \
      action.AutocorrFront(), action.AutocorrBack());
  printf("spin: %.12e %.12e %.4f %.4f\n", \
      m_mean, m_err, \
      spin.AutocorrFront(), spin.AutocorrBack());
  printf("m^2: %.12e %.12e %.4f %.4f\n", \
      m2_mean, m2_err, \
      mag_2.AutocorrFront(), mag_2.AutocorrBack());
  printf("m^4: %.12e %.12e %.4f %.4f\n", \
      m4_mean, m4_err, \
      mag_4.AutocorrFront(), mag_4.AutocorrBack());

  double U4_mean = 1.5 * (1.0 - m4_mean / (3.0 * m2_mean * m2_mean));
  double U4_err = 0.5 * U4_mean * sqrt(pow(m4_err / m4_mean, 2.0) \
      + pow(2.0 * m2_err / m2_mean, 2.0));
  printf("U4: %.12e %.12e\n", U4_mean, U4_err);

  double m_susc_mean = (m2_mean - m_mean * m_mean) / vol;
  double m_susc_err = sqrt(pow(m2_err, 2.0) \
      + pow(2.0 * m_mean * m_err, 2.0)) / vol;
  printf("m_susc: %.12e %.12e\n", m_susc_mean, m_susc_err);

  // open an output file
  char run_id[50];
  char path[200];
  sprintf(run_id, "%d_%d", n_refine, q);
  FILE* file;

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
    fprintf(file, "%04d %.16e %.16e %d %.16e %.16e\n", l, \
        legendre_2pt[l].Mean(), legendre_2pt[l].Error(), \
        legendre_2pt[l].n, legendre_2pt[l].sum, legendre_2pt[l].sum2);
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
    fprintf(file, "%04d %.16e %.16e %d %.16e %.16e\n", l, \
        legendre_4pt[l].Mean(), legendre_4pt[l].Error(), \
        legendre_4pt[l].n, legendre_4pt[l].sum, legendre_4pt[l].sum2);
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
    fprintf(file, "%04d %.16e %.16e %d %.16e %.16e\n", ylm_i, \
        ylm_2pt[ylm_i].Mean(), ylm_2pt[ylm_i].Error(), \
        ylm_2pt[ylm_i].n, ylm_2pt[ylm_i].sum, ylm_2pt[ylm_i].sum2);
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
    fprintf(file, "%04d %.16e %.16e %d %.16e %.16e\n", ylm_i, \
        ylm_4pt[ylm_i].Mean(), ylm_4pt[ylm_i].Error(), \
        ylm_4pt[ylm_i].n, ylm_4pt[ylm_i].sum, ylm_4pt[ylm_i].sum2);
    m++;
    if (m > l) {
      l++;
      m = 0;
    }
  }
  fclose(file);

  return 0;
}
