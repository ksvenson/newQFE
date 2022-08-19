// s3_lap_dec.cc

#include <cstdio>
#include <iostream>
#include "s3.h"

// calculate DEC Laplacian weights and eigenvalues in hyperspherical harmonic basis

typedef std::complex<double> Complex;
typedef Eigen::Vector4<double> Vec4;

int main(int argc, char* argv[]) {

  assert(argc > 1);
  char* base_path = argv[1];

  QfeLatticeS3 lattice(0);

  // read lattice
  char lattice_path[200];
  sprintf(lattice_path, "%s.dat", base_path);
  FILE* lattice_file = fopen(lattice_path, "r");
  assert(lattice_file != nullptr);
  lattice.ReadLattice(lattice_file);
  fclose(lattice_file);

  // set site FEM weights to zero
  for (int s = 0; s < lattice.n_sites; s++) {
    lattice.sites[s].wt = 0.0;
  }

  // set link FEM weights to zero
  for (int l = 0; l < lattice.n_links; l++) {
    lattice.links[l].wt = 0.0;
  }

  // compute the DEC laplacian
  for (int c = 0; c < lattice.n_cells; c++) {

    // coordinates of vertices
    Vec4 cell_r[5];
    cell_r[0] = Vec4::Zero();  // distance from origin to each vertex is 1
    cell_r[1] = lattice.r[lattice.cells[c].sites[0]];
    cell_r[2] = lattice.r[lattice.cells[c].sites[1]];
    cell_r[3] = lattice.r[lattice.cells[c].sites[2]];
    cell_r[4] = lattice.r[lattice.cells[c].sites[3]];

    // generate the Cayley-Menger matrix
    Eigen::Matrix<double, 5, 5> CM;
    for (int i = 0; i < 5; i++) {
      CM(i,i) = 0.0;
      for (int j = i + 1; j < 5; j++) {
        CM(i,j) = (cell_r[i] - cell_r[j]).squaredNorm();
        CM(j,i) = CM(i,j);
      }
    }

    Eigen::Vector<double, 5> cell_lhs(1.0, 0.0, 0.0, 0.0, 0.0);
    Eigen::Vector<double, 5> cell_xi = CM.inverse() * cell_lhs;
    double cell_cr_sq = -cell_xi(0) / 2.0;

    // i and j are 1-indexed in the Cayley-Menger matrix
    for (int i = 1; i <= 4; i++) {
      for (int j = i + 1; j <= 4; j++) {

        // find the other two corners
        int k = 1;
        while (k == i || k == j) k++;
        int l = k + 1;
        while (l == i || l == j) l++;

        double x_ijk = 2.0 * (CM(i,j) * CM(i,k) + CM(i,j) * CM(j,k) + CM(i,k) * CM(j,k))
            - (CM(i,j) * CM(i,j) + CM(i,k) * CM(i,k) + CM(j,k) * CM(j,k));
        double x_ijl = 2.0 * (CM(i,j) * CM(i,l) + CM(i,j) * CM(j,l) + CM(i,l) * CM(j,l))
            - (CM(i,j) * CM(i,j) + CM(i,l) * CM(i,l) + CM(j,l) * CM(j,l));

        double A_tri_ijk = 0.25 * sqrt(x_ijk);
        double A_tri_ijl = 0.25 * sqrt(x_ijl);
        double dual_ijk = CM(i,k) + CM(j,k) - CM(i,j);
        double dual_ijl = CM(i,l) + CM(j,l) - CM(i,j);

        double h_ijk = sqrt(cell_cr_sq - CM(i,j) * CM(i,k) * CM(j,k) / x_ijk);
        double h_ijl = sqrt(cell_cr_sq - CM(i,j) * CM(i,l) * CM(j,l) / x_ijl);

        // sign factor
        if (cell_xi(l) < 0.0) h_ijk *= -1.0;
        if (cell_xi(k) < 0.0) h_ijl *= -1.0;

        double wt = (dual_ijk * h_ijk / A_tri_ijk + dual_ijl * h_ijl / A_tri_ijl) / 16.0;
        int s_i = lattice.cells[c].sites[i - 1];
        int s_j = lattice.cells[c].sites[j - 1];
        int l_ij = lattice.FindLink(s_i, s_j);

        // set FEM weights
        lattice.links[l_ij].wt += wt;
        lattice.sites[s_i].wt += wt * CM(i,j) / 6.0;
        lattice.sites[s_j].wt += wt * CM(i,j) / 6.0;
      }
    }
  }

  // lattice.PrintSites();
  // lattice.PrintLinks();

  double site_vol = 0.0;
  for (int s = 0; s < lattice.n_sites; s++) {
    site_vol += lattice.sites[s].wt;
  }
  printf("site_vol: %.12f\n", site_vol);

  double cell_vol = 0.0;
  double cr_sum = 0.0;
  for (int c = 0; c < lattice.n_cells; c++) {
    lattice.cells[c].wt = lattice.CellVolume(c);
    cell_vol += lattice.cells[c].wt;
    Vec4 cell_cc = lattice.CellCircumcenter(c);
    double cr = (cell_cc - lattice.r[lattice.cells[c].sites[0]]).norm();
    cr_sum += cr;
  }
  double cr_mean = cr_sum / double(lattice.n_cells);
  printf("cell_vol: %.12f\n", cell_vol);
  printf("cr_mean: %.12f\n", cr_mean);

  // make sure cell volume matches site volume
  if (isnan(site_vol) || isnan(cell_vol)) exit(1);
  if (fabs(site_vol - cell_vol) > 1.0e-4) exit(1);

  // normalize site volume
  lattice.vol = double(lattice.n_sites);
  double site_norm = site_vol / lattice.vol;
  for (int s = 0; s < lattice.n_sites; s++) {
    lattice.sites[s].wt /= site_norm;
  }

  // normalize link weights
  double link_norm = cbrt(site_norm);
  for (int l = 0; l < lattice.n_links; l++) {
    lattice.links[l].wt /= link_norm;
  }

  int j_max = 12;

  // check integrator
  char int_path[200];
  sprintf(int_path, "%s_int.dat", base_path);
  FILE* int_file = fopen(int_path, "w");
  for (int j = 0; j <= j_max; j++) {
    for (int l = 0; l <= j; l++) {
      for (int m = 0; m <= l; m++) {
        Complex yjlm_sum = 0.0;
        for (int s = 0; s < lattice.n_sites; s++) {
          Complex y = lattice.GetYjlm(s, j, l, m);
          double wt = lattice.sites[s].wt;
          yjlm_sum += wt * y;
        }
        Complex yjlm_mean = yjlm_sum * sqrt(2.0 * M_PI * M_PI) / lattice.vol;
        if (std::abs(yjlm_mean) < 1.0e-10) continue;

        fprintf(int_file, "%02d %02d %02d %+.12e %+.12e\n", j, l, m, real(yjlm_mean), imag(yjlm_mean));
      }
    }
  }
  fclose(int_file);

  j_max = 12;

  // estimate the Laplacian eigenvalues in the hyperspherical harmonic basis
  char lap_path[200];
  sprintf(lap_path, "%s_lap.dat", base_path);
  FILE* lap_file = fopen(lap_path, "w");
  for (int j = 0; j <= j_max; j++) {
    for (int l = 0; l <= j; l++) {
      for (int m = 0; m <= l; m++) {
        double yjlm_sum = 0.0;
        for (int link = 0; link < lattice.n_links; link++) {
          int s_a = lattice.links[link].sites[0];
          int s_b = lattice.links[link].sites[1];
          Complex y_a = lattice.GetYjlm(s_a, j, l, m);
          Complex y_b = lattice.GetYjlm(s_b, j, l, m);
          double wt = lattice.links[link].wt;
          yjlm_sum += wt * std::norm(y_a - y_b);
        }
        double yjlm_mean = yjlm_sum * cbrt(2.0 * M_PI * M_PI / lattice.vol);

        fprintf(lap_file, "%02d %02d %02d %+.12e\n", j, l, m, yjlm_mean);
      }
    }
  }
  fclose(lap_file);

  return 0;
}
