// s2_lap_dec.cc

#include <cstdio>
#include <iostream>

#include "s2.h"

// calculate DEC Laplacian weights and eigenvalues in spherical harmonic basis

typedef std::complex<double> Complex;
typedef Eigen::Vector3<double> Vec3;

int main(int argc, char* argv[]) {
  int q = 5;
  int k = 1;
  std::string orbit_path = "";

  if (argc > 1) q = atoi(argv[1]);
  if (argc > 2) k = atoi(argv[2]);
  if (argc > 3) orbit_path = argv[3];

  QfeLatticeS2 lattice(q, k);

  if (!orbit_path.empty()) {
    FILE* orbit_file = fopen(orbit_path.c_str(), "r");
    assert(orbit_file != nullptr);
    lattice.ReadOrbits(orbit_file);
    fclose(orbit_file);
  }

  lattice.UpdateWeights();

  double a_lat = lattice.CalcLatticeSpacing();
  double site_vol = 0.0;
  for (int s = 0; s < lattice.n_sites; s++) {
    site_vol += lattice.sites[s].wt * a_lat * a_lat;
  }

  double link_vol = 0.0;
  for (int l = 0; l < lattice.n_links; l++) {
    link_vol += 0.5 * lattice.links[l].wt * lattice.EdgeSquared(l);
  }

  double face_vol = 0.0;
  for (int f = 0; f < lattice.n_faces; f++) {
    lattice.faces[f].wt = lattice.FlatArea(f);
    face_vol += lattice.faces[f].wt;
  }

  printf("site_vol: %.12f\n", site_vol);
  printf("link_vol: %.12f\n", link_vol);
  printf("face_vol: %.12f\n", face_vol);

  int l_max = 24;

  // check integrator
  // char int_path[200];
  // sprintf(int_path, "%s_dec_int.dat", base_path);
  // FILE* int_file = fopen(int_path, "w");
  for (int l = 0; l <= l_max; l++) {
    for (int m = 0; m <= l; m++) {
      Complex ylm_sum = 0.0;
      for (int s = 0; s < lattice.n_sites; s++) {
        Complex y = lattice.CalcYlm(s, l, m);
        double wt = lattice.sites[s].wt;
        ylm_sum += wt * y;
      }
      Complex ylm_mean = ylm_sum * sqrt(4.0 * M_PI) / lattice.vol;
      if (std::abs(ylm_mean) < 1.0e-10) continue;

      printf("%02d %02d %+.12e %+.12e\n", l, m, real(ylm_mean), imag(ylm_mean));
      // fprintf(int_file, "%02d %02d %+.12e %+.12e\n", l, m, real(ylm_mean),
      // imag(ylm_mean));
    }
  }
  // fclose(int_file);

  // estimate the Laplacian eigenvalues in the hyperspherical harmonic basis
  // char lap_path[200];
  // sprintf(lap_path, "%s_dec_lap.dat", base_path);
  // FILE* lap_file = fopen(lap_path, "w");
  for (int l = 0; l <= l_max; l++) {
    for (int m = 0; m <= l; m++) {
      double ylm_sum = 0.0;
      for (int link = 0; link < lattice.n_links; link++) {
        int s_a = lattice.links[link].sites[0];
        int s_b = lattice.links[link].sites[1];
        Complex y_a = lattice.CalcYlm(s_a, l, m);
        Complex y_b = lattice.CalcYlm(s_b, l, m);
        double wt = lattice.links[link].wt;
        ylm_sum += wt * std::norm(y_a - y_b);
      }

      printf("%02d %02d %.12f\n", l, m, Chop(ylm_sum));
      // fprintf(lap_file, "%02d %02d %+.12f\n", l, m, ylm_sum);
    }
  }
  // fclose(lap_file);

  // // write the lattice to file
  // sprintf(lattice_path, "%s_dec.dat", base_path);
  // lattice_file = fopen(lattice_path, "w");
  // lattice.WriteLattice(lattice_file);

  return 0;
}
