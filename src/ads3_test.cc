// ads3_test.cc

#include <cstdio>
#include "ads3.h"
#include "phi4.h"
#include "statistics.h"

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

  int N = atoi(argv[1]);
  printf("N: %d\n", N);

  int q = 7;
  printf("q: %d\n", q);

  int Nt = atoi(argv[2]);
  printf("Nt: %d\n", Nt);

  double musq = std::stod(argv[3]);
  printf("musq: %.4f\n", musq);

  double lambda = std::stod(argv[4]);
  printf("lambda: %.4f\n", lambda);

  QfeLatticeAdS3 lattice(N, q, Nt);
  printf("total sites: %d\n", lattice.n_sites + lattice.n_dummy);
  printf("bulk sites: %d\n", lattice.n_bulk);
  printf("boundary sites: %d\n", lattice.n_boundary);
  printf("dummy sites: %d\n", lattice.n_dummy);

  printf("average rho/cosh(rho) at each level:\n");
  for (int n = 0; n < N; n++) {
    printf("%d %.12f %.12f\n", \
        n, lattice.level_rho[n], lattice.level_cosh_rho[n]);
  }

  QfePhi4 field(&lattice, musq, lambda);
  field.ColdStart();
  field.metropolis_z = 0.1;

  printf("initial action: %.12f\n", field.Action());

  // measurements
  std::vector<double> mag;
  std::vector<double> mag_bulk;
  std::vector<double> mag_boundary;
  std::vector<double> action;
  std::vector<double> cluster_size;
  std::vector<double> accept_metropolis;
  std::vector<double> accept_overrelax;
  std::vector<double> demon;

  int n_therm = 1000;
  int n_traj = 20000;
  int n_skip = 20;
  int n_wolff = 5;
  int n_metropolis = 1;
  for (int n = 0; n < (n_traj + n_therm); n++) {

    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      cluster_size_sum += field.WolffUpdate();
    }
    double metropolis_sum = 0.0;
    for (int j = 0; j < n_metropolis; j++) {
      metropolis_sum += field.Metropolis();
    }
    cluster_size.push_back(double(cluster_size_sum) / double(lattice.n_sites));
    accept_metropolis.push_back(metropolis_sum);
    accept_overrelax.push_back(field.Overrelax());
    demon.push_back(field.overrelax_demon);

    if (n % n_skip || n < n_therm) continue;

    // measure <phi> in the bulk
    double phi_bulk_sum = 0.0;
    int n_bulk_sum = 0;
    for (int level = 0; level < N - 1; level++) {
      for (int i = 0; i < lattice.level_sites[level].size(); i++) {
        int s = lattice.level_sites[level][i];
        phi_bulk_sum += field.phi[s] * lattice.sites[s].wt;
        n_bulk_sum++;
      }
    }
    mag_bulk.push_back(phi_bulk_sum / double(n_bulk_sum));

    // measure <phi> on the boundary
    double phi_boundary_sum = 0.0;
    int n_boundary_sum = 0;
    for (int i = 0; i < lattice.level_sites[N].size(); i++) {
      int s = lattice.level_sites[N][i];
      phi_boundary_sum += field.phi[s] * lattice.sites[s].wt;
      n_boundary_sum++;
    }
    mag_boundary.push_back(phi_boundary_sum / double(n_boundary_sum));

    action.push_back(field.Action());
    mag.push_back(field.MeanPhi());
    printf("%06d %.12f %+.12f %.4f %.4f %.12f %d\n", \
        n, action.back(), mag.back(), \
        accept_metropolis.back(), \
        accept_overrelax.back(), demon.back(), \
        cluster_size_sum);
  }

  std::vector<double> mag_abs(mag.size());
  std::vector<double> mag2(mag.size());
  std::vector<double> mag4(mag.size());
  for (int i = 0; i < mag.size(); i++) {
    double m = mag[i];
    double m2 = m * m;
    mag_abs[i] = abs(m);
    mag2[i] = m2;
    mag4[i] = m2 * m2;
  }

  std::vector<double> mag_abs_bulk(mag_bulk.size());
  std::vector<double> mag2_bulk(mag_bulk.size());
  std::vector<double> mag4_bulk(mag_bulk.size());
  for (int i = 0; i < mag_bulk.size(); i++) {
    double m = mag_bulk[i];
    double m2 = m * m;
    mag_abs_bulk[i] = abs(m);
    mag2_bulk[i] = m2;
    mag4_bulk[i] = m2 * m2;
  }

  std::vector<double> mag_abs_boundary(mag_boundary.size());
  std::vector<double> mag2_boundary(mag_boundary.size());
  std::vector<double> mag4_boundary(mag_boundary.size());
  for (int i = 0; i < mag_boundary.size(); i++) {
    double m = mag_boundary[i];
    double m2 = m * m;
    mag_abs_boundary[i] = abs(m);
    mag2_boundary[i] = m2;
    mag4_boundary[i] = m2 * m2;
  }

  printf("cluster_size/V: %.4f\n", Mean(cluster_size));
  printf("accept_metropolis: %.4f\n", Mean(accept_metropolis));
  printf("accept_overrelax: %.4f\n", Mean(accept_overrelax));
  printf("demon: %.12f (%.12f)\n", Mean(demon), JackknifeMean(demon));
  printf("action: %.12e (%.12e), %.4f\n", \
      Mean(action), JackknifeMean(action), AutocorrTime(action));

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

  char path[50];
  FILE* file;

  sprintf(path, "all_%d.dat", N);
  file = fopen(path, "a");
  fprintf(file, "%.12f", musq);
  fprintf(file, " %.12f", lambda);
  fprintf(file, " %.12e %.12e", \
      U4(mag2, mag4), \
      JackknifeU4(mag2, mag4));
  fprintf(file, " %.12e %.12e", \
      Susceptibility(mag2, mag_abs), \
      JackknifeSusceptibility(mag2, mag_abs));
  fprintf(file, "\n");
  fclose(file);

  sprintf(path, "bulk_%d.dat", N);
  file = fopen(path, "a");
  fprintf(file, "%.12f", musq);
  fprintf(file, " %.12f", lambda);
  fprintf(file, " %.12e %.12e", \
      U4(mag2_bulk, mag4_bulk), \
      JackknifeU4(mag2_bulk, mag4_bulk));
  fprintf(file, " %.12e %.12e", \
      Susceptibility(mag2_bulk, mag_abs_bulk), \
      JackknifeSusceptibility(mag2_bulk, mag_abs_bulk));
  fprintf(file, "\n");
  fclose(file);

  sprintf(path, "boundary_%d.dat", N);
  file = fopen(path, "a");
  fprintf(file, "%.12f", musq);
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
