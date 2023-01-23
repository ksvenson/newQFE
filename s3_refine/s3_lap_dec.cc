// s3_lap_dec.cc

#include <cstdio>
#include <iostream>

#include "s3.h"

// calculate DEC Laplacian weights and eigenvalues in hyperspherical harmonic
// basis

int main(int argc, char* argv[]) {
  int q = 5;
  int k = 2;

  if (argc > 1) q = atoi(argv[1]);
  if (argc > 2) k = atoi(argv[2]);

  QfeLatticeS3 lattice(q, k);
  lattice.Inflate();
  lattice.CalcFEMWeights();

  int j_max = 12;

  // check integrator
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

        printf("%02d %02d %02d %+.12e %+.12e\n", j, l, m, real(yjlm_mean),
               imag(yjlm_mean));
      }
    }
  }

  j_max = 12;

  // estimate the Laplacian eigenvalues in the hyperspherical harmonic basis
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

        printf("%02d %02d %02d %+.12f\n", j, l, m, yjlm_mean);
      }
    }
  }

  return 0;
}
