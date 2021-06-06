// ads3_crit.cc

#include <cstdio>
#include <vector>
#include <string>
#include <getopt.h>
#include "ads3.h"
#include "phi4.h"
#include "statistics.h"

using std::vector;
using std::stod;

int main(int argc, char* argv[]) {

  // choose number of time slices to keep Nt / cosh(rho[boundary]) constant
  // n_layers = 4, Nt = 8 is the base case

  // n_layers      Nt        cosh(rho)_boundary         Nt / cosh(rho)
  // --------     ----       ------------------         --------------
  //    3            3           10.22132448              0.293504037
  //   *4*           8           27.07811265              0.295441566
  //    5           21           71.89717658              0.292083792
  //    6           56          190.9628102               0.293250817
  //    7          150          507.2320663               0.295722629
  //    8          398         1347.310396                0.295403347

  int n_layers = 4;
  int q = 7;
  int Nt = 8;
  double musq = 1.0;
  double lambda = 1.0;
  int n_therm = 1000;
  int n_traj = 20000;
  int n_skip = 20;
  int n_wolff = 4;
  int n_metropolis = 1;
  double metropolis_z = 0.1;

  while (true) {

    struct option long_options[] = {
      {"n_layers", required_argument, 0, 'N'},
      {"q", required_argument, 0, 'q'},
      {"n_t", required_argument, 0, 'T'},
      {"musq", required_argument, 0, 'm'},
      {"lambda", required_argument, 0, 'l'},
      {"n_therm", required_argument, 0, 'h'},
      {"n_traj", required_argument, 0, 't'},
      {"n_skip", required_argument, 0, 's'},
      {"n_wolff", required_argument, 0, 'w'},
      {"n_metropolis", required_argument, 0, 'e'},
      {"metropolis_z", required_argument, 0, 'z'},
      {0, 0, 0, 0}
    };

    const char* short_options = "NqTmlhtswez";

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N':
        n_layers = atoi(optarg);
        break;
      case 'q':
        q = stod(optarg);
        break;
      case 'T':
        Nt = atoi(optarg);
        break;
      case 'm':
        musq = stod(optarg);
        break;
      case 'l':
        lambda = stod(optarg);
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
        metropolis_z = stod(optarg);
        break;
      default:
        break;
    }
  }

  printf("n_layers: %d\n", n_layers);
  printf("q: %d\n", q);
  printf("Nt: %d\n", Nt);
  printf("musq: %.4f\n", musq);
  printf("lambda: %.4f\n", lambda);
  printf("n_therm: %d\n", n_therm);
  printf("n_traj: %d\n", n_traj);
  printf("n_skip: %d\n", n_skip);
  printf("n_wolff: %d\n", n_wolff);
  printf("n_metropolis: %d\n", n_metropolis);
  printf("metropolis_z: %.4f\n", metropolis_z);

  QfeLatticeAdS3 lattice(n_layers, q, Nt);
  printf("total sites: %d\n", lattice.n_sites + lattice.n_dummy);
  printf("bulk sites: %d\n", lattice.n_bulk);
  printf("boundary sites: %d\n", lattice.n_boundary);
  printf("dummy sites: %d\n", lattice.n_dummy);
  printf("t_scale: %.12f\n", lattice.t_scale);

  printf("average rho/cosh(rho) at each layer:\n");
  for (int n = 0; n <= n_layers + 1; n++) {
    printf("%d %.12f %.12f %.12f\n", n, \
        lattice.layer_rho[n], \
        lattice.layer_cosh_rho[n], \
        lattice.total_cosh_rho[n]);
  }

  QfePhi4 field(&lattice, musq, lambda);
  field.metropolis_z = metropolis_z;
  field.HotStart();

  printf("initial action: %.12f\n", field.Action());

  // measurements
  vector<double> mag;
  vector<double> mag_bulk;
  vector<double> mag_boundary;
  vector<double> action;
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
    cluster_size.Measure(double(cluster_size_sum) / double(lattice.n_sites));
    accept_metropolis.Measure(metropolis_sum);
    accept_overrelax.Measure(field.Overrelax());

    if (n % n_skip || n < n_therm) continue;

    demon.Measure(field.overrelax_demon);

    // measure <phi> in the bulk
    double phi_bulk_sum = 0.0;
    int n_bulk_sum = 0;
    for (int layer = 0; layer < (n_layers - 1); layer++) {
      for (int i = 0; i < lattice.layer_sites[layer].size(); i++) {
        int s = lattice.layer_sites[layer][i];
        phi_bulk_sum += field.phi[s] * lattice.sites[s].wt;
        n_bulk_sum++;
      }
    }
    mag_bulk.push_back(phi_bulk_sum / double(n_bulk_sum));

    // measure <phi> on the boundary
    double phi_boundary_sum = 0.0;
    int n_boundary_sum = 0;
    for (int i = 0; i < lattice.layer_sites[n_layers].size(); i++) {
      int s = lattice.layer_sites[n_layers][i];
      phi_boundary_sum += field.phi[s] * lattice.sites[s].wt;
      n_boundary_sum++;
    }
    mag_boundary.push_back(phi_boundary_sum / double(n_boundary_sum));

    action.push_back(field.Action());
    mag.push_back(field.MeanPhi());
    printf("%06d %.12f %+.12f %.4f %.4f %.12f %.4f\n", \
        n, action.back(), mag.back(), \
        accept_metropolis.last, \
        accept_overrelax.last, demon.last, \
        cluster_size.last);
  }

  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());
  printf("accept_overrelax: %.4f\n", accept_overrelax.Mean());
  printf("demon: %.12f (%.12f)\n", demon.Mean(), demon.Error());
  printf("action: %.12e (%.12e), %.4f\n", \
      Mean(action), JackknifeMean(action), AutocorrTime(action));

  vector<double> mag_abs(mag.size());
  vector<double> mag2(mag.size());
  vector<double> mag4(mag.size());
  for (int i = 0; i < mag.size(); i++) {
    double m = mag[i];
    double m2 = m * m;
    mag_abs[i] = abs(m);
    mag2[i] = m2;
    mag4[i] = m2 * m2;
  }

  vector<double> mag_abs_bulk(mag_bulk.size());
  vector<double> mag2_bulk(mag_bulk.size());
  vector<double> mag4_bulk(mag_bulk.size());
  for (int i = 0; i < mag_bulk.size(); i++) {
    double m = mag_bulk[i];
    double m2 = m * m;
    mag_abs_bulk[i] = abs(m);
    mag2_bulk[i] = m2;
    mag4_bulk[i] = m2 * m2;
  }

  vector<double> mag_abs_boundary(mag_boundary.size());
  vector<double> mag2_boundary(mag_boundary.size());
  vector<double> mag4_boundary(mag_boundary.size());
  for (int i = 0; i < mag_boundary.size(); i++) {
    double m = mag_boundary[i];
    double m2 = m * m;
    mag_abs_boundary[i] = abs(m);
    mag2_boundary[i] = m2;
    mag4_boundary[i] = m2 * m2;
  }

  printf("\nbulk + boundary:\n");
  printf("m: %.12e (%.12e), %.4f\n", \
      Mean(mag), JackknifeMean(mag), AutocorrTime(mag));
  printf("m^2: %.12e (%.12e), %.4f\n", \
      Mean(mag2), JackknifeMean(mag2), AutocorrTime(mag2));
  printf("m^4: %.12e (%.12e), %.4f\n", \
      Mean(mag4), JackknifeMean(mag4), AutocorrTime(mag4));
  printf("U4: %.12e (%.12e)\n", U4(mag2, mag4), JackknifeU4(mag2, mag4));
  printf("susceptibility: %.12e (%.12e)\n", Susceptibility(mag2, mag_abs), \
      JackknifeSusceptibility(mag2, mag_abs));

  printf("\nbulk:\n");
  printf("m: %.12e (%.12e), %.4f\n", \
      Mean(mag_bulk), JackknifeMean(mag_bulk), AutocorrTime(mag_bulk));
  printf("m^2: %.12e (%.12e), %.4f\n", \
      Mean(mag2_bulk), JackknifeMean(mag2_bulk), AutocorrTime(mag2_bulk));
  printf("m^4: %.12e (%.12e), %.4f\n", \
      Mean(mag4_bulk), JackknifeMean(mag4_bulk), AutocorrTime(mag4_bulk));
  printf("U4: %.12e (%.12e)\n", \
      U4(mag2_bulk, mag4_bulk), JackknifeU4(mag2_bulk, mag4_bulk));
  printf("susceptibility: %.12e (%.12e)\n", \
      Susceptibility(mag2_bulk, mag_abs_bulk), \
      JackknifeSusceptibility(mag2_bulk, mag_abs_bulk));

  printf("\nboundary:\n");
  printf("m: %.12e (%.12e), %.4f\n", \
      Mean(mag_boundary), JackknifeMean(mag_boundary), \
      AutocorrTime(mag_boundary));
  printf("m^2: %.12e (%.12e), %.4f\n", \
      Mean(mag2_boundary), JackknifeMean(mag2_boundary), \
      AutocorrTime(mag2_boundary));
  printf("m^4: %.12e (%.12e), %.4f\n", \
      Mean(mag4_boundary), JackknifeMean(mag4_boundary), \
      AutocorrTime(mag4_boundary));
  printf("U4: %.12e (%.12e)\n", \
      U4(mag2_boundary, mag4_boundary), \
      JackknifeU4(mag2_boundary, mag4_boundary));
  printf("susceptibility: %.12e (%.12e)\n", \
      Susceptibility(mag2_boundary, mag_abs_boundary), \
      JackknifeSusceptibility(mag2_boundary, mag_abs_boundary));

  FILE* file;

  file = fopen("ads3_crit_all.dat", "a");
  fprintf(file, "%d", n_layers);
  fprintf(file, " %d", Nt);
  fprintf(file, " %.12f", musq);
  fprintf(file, " %.12f", lambda);
  fprintf(file, " %.12e %.12e", \
      U4(mag2, mag4), \
      JackknifeU4(mag2, mag4));
  fprintf(file, " %.12e %.12e", \
      Susceptibility(mag2, mag_abs), \
      JackknifeSusceptibility(mag2, mag_abs));
  fprintf(file, "\n");
  fclose(file);

  file = fopen("ads3_crit_bulk.dat", "a");
  fprintf(file, "%d", n_layers);
  fprintf(file, " %d", Nt);
  fprintf(file, " %.12f", musq);
  fprintf(file, " %.12f", lambda);
  fprintf(file, " %.12e %.12e", \
      U4(mag2_bulk, mag4_bulk), \
      JackknifeU4(mag2_bulk, mag4_bulk));
  fprintf(file, " %.12e %.12e", \
      Susceptibility(mag2_bulk, mag_abs_bulk), \
      JackknifeSusceptibility(mag2_bulk, mag_abs_bulk));
  fprintf(file, "\n");
  fclose(file);

  file = fopen("ads3_crit_boundary.dat", "a");
  fprintf(file, "%d", n_layers);
  fprintf(file, " %d", Nt);
  fprintf(file, " %.12f", musq);
  fprintf(file, " %.12f", lambda);
  fprintf(file, " %.12e %.12e", \
      U4(mag2_boundary, mag4_boundary), \
      JackknifeU4(mag2_boundary, mag4_boundary));
  fprintf(file, " %.12e %.12e", \
      Susceptibility(mag2_boundary, mag_abs_boundary), \
      JackknifeSusceptibility(mag2_boundary, mag_abs_boundary));
  fprintf(file, "\n");
  fclose(file);

  return 0;
}
