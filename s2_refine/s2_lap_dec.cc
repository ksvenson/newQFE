// s2_lap_dec.cc

#include <cstdio>
#include <iostream>

#include "s2.h"

// calculate DEC Laplacian weights and eigenvalues in spherical harmonic basis

typedef std::complex<double> Complex;
typedef Eigen::Vector3<double> Vec3;

int main(int argc, char* argv[]) {
  assert(argc > 1);
  char* base_path = argv[1];

  QfeLatticeS2 lattice(0);

  // read lattice
  char lattice_path[200];
  sprintf(lattice_path, "%s.dat", base_path);
  FILE* lattice_file = fopen(lattice_path, "r");
  assert(lattice_file != nullptr);
  lattice.ReadLattice(lattice_file);
  fclose(lattice_file);

  // set site weights to zero
  for (int s = 0; s < lattice.n_sites; s++) {
    lattice.sites[s].wt = 0.0;
  }

  // loop over links to update weights
  for (int l = 0; l < lattice.n_links; l++) {
    lattice.links[l].wt = 0.0;
    for (int i = 0; i < 2; i++) {
      // find the other two edges of this face
      int f = lattice.links[l].faces[i];
      int e = 0;
      while (lattice.faces[f].edges[e] != l) e++;
      int e1 = (e + 1) % 3;
      int e2 = (e + 2) % 3;
      int l1 = lattice.faces[f].edges[e1];
      int l2 = lattice.faces[f].edges[e2];

      // find the area associated with this face
      double sq_edge = lattice.EdgeSquared(l);
      double sq_edge_1 = lattice.EdgeSquared(l1);
      double sq_edge_2 = lattice.EdgeSquared(l2);
      double tri_area = lattice.FlatArea(f);
      double half_wt = (sq_edge_1 + sq_edge_2 - sq_edge) / (8.0 * tri_area);
      lattice.links[l].wt += half_wt;

      // add to the weights of the two sites connected by this link
      int s_a = lattice.links[l].sites[0];
      int s_b = lattice.links[l].sites[1];
      lattice.sites[s_a].wt += 0.25 * half_wt * sq_edge;
      lattice.sites[s_b].wt += 0.25 * half_wt * sq_edge;
    }
  }

  double site_vol = 0.0;
  for (int s = 0; s < lattice.n_sites; s++) {
    site_vol += lattice.sites[s].wt;
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

  // normalize site weights to 1
  lattice.vol = double(lattice.n_sites);
  double site_norm = site_vol / lattice.vol;
  for (int s = 0; s < lattice.n_sites; s++) {
    lattice.sites[s].wt /= site_norm;
  }

  // normalize face weights
  for (int f = 0; f < lattice.n_faces; f++) {
    lattice.faces[f].wt /= site_norm;
  }

  int l_max = 12;
  lattice.UpdateYlm(l_max);

  // check integrator
  char int_path[200];
  sprintf(int_path, "%s_dec_int.dat", base_path);
  FILE* int_file = fopen(int_path, "w");
  for (int l = 0; l <= l_max; l++) {
    for (int m = 0; m <= l; m++) {
      Complex ylm_sum = 0.0;
      for (int s = 0; s < lattice.n_sites; s++) {
        Complex y = lattice.GetYlm(s, l, m);
        double wt = lattice.sites[s].wt;
        ylm_sum += wt * y;
      }
      Complex ylm_mean = ylm_sum * sqrt(4.0 * M_PI) / lattice.vol;
      if (std::abs(ylm_mean) < 1.0e-10) continue;

      fprintf(int_file, "%02d %02d %+.12e %+.12e\n", l, m, real(ylm_mean),
              imag(ylm_mean));
    }
  }
  fclose(int_file);

  // estimate the Laplacian eigenvalues in the hyperspherical harmonic basis
  char lap_path[200];
  sprintf(lap_path, "%s_dec_lap.dat", base_path);
  FILE* lap_file = fopen(lap_path, "w");
  for (int l = 0; l <= l_max; l++) {
    for (int m = 0; m <= l; m++) {
      double ylm_sum = 0.0;
      for (int link = 0; link < lattice.n_links; link++) {
        int s_a = lattice.links[link].sites[0];
        int s_b = lattice.links[link].sites[1];
        Complex y_a = lattice.GetYlm(s_a, l, m);
        Complex y_b = lattice.GetYlm(s_b, l, m);
        double wt = lattice.links[link].wt;
        ylm_sum += wt * std::norm(y_a - y_b);
      }

      fprintf(lap_file, "%02d %02d %+.12e\n", l, m, ylm_sum);
    }
  }
  fclose(lap_file);

  // write the lattice to file
  sprintf(lattice_path, "%s_dec.dat", base_path);
  lattice_file = fopen(lattice_path, "w");
  lattice.WriteLattice(lattice_file);

  return 0;
}
