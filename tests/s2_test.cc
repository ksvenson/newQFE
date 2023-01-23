// s2_test.cc

#include "s2.h"

#include <cstdio>

int main(int argc, char* argv[]) {
  // list of refinement values to test
  int k_list[] = {1,  2,  3,  4,  5,  6,  7,  8,   10,  12,  16,  20,  24,
                  28, 32, 40, 48, 56, 64, 96, 128, 144, 192, 256, 384, 512};
  int n_k = sizeof(k_list) / sizeof(int);

  for (int q = 3; q <= 5; q++) {
    for (int i_k = 0; i_k < n_k; i_k++) {
      int k = k_list[i_k];
      printf("Testing q%dk%d lattice\n", q, k);
      QfeLatticeS2 lattice(q, k);

      // check Euler characteristic
      int euler_chi = lattice.n_sites - lattice.n_links + lattice.n_faces;
      // printf("euler characteristic: %d\n", euler_chi);
      assert(euler_chi == 2);

      // check that each site has the proper number of neighbors
      for (int s = 0; s < lattice.n_sites; s++) {
        if (lattice.site_orbit[s] == 0) {
          assert(lattice.sites[s].nn == q);
        } else {
          assert(lattice.sites[s].nn == 6);
        }
      }

      // check antipodal points
      if (q != 3) lattice.UpdateAntipodes();

      // calculate DEC weights
      lattice.UpdateWeights();

      // check that low ell irreps of O(3) can be integrated exactly
      double error_tol = 1.0e-12;
      int l_max = (q == 5 ? 2 : 1);
      for (int l = 0; l <= l_max; l++) {
        for (int m = -l; m <= l; m++) {
          Complex ylm_sum = 0.0;
          for (int s = 0; s < lattice.n_sites; s++) {
            ylm_sum += lattice.sites[s].wt * lattice.CalcYlm(s, l, m);
          }
          Complex ylm_mean = ylm_sum * sqrt(4.0 * M_PI) / lattice.vol;
          // printf("l,m=%d,%+d: %+.12e %+.12e\n", l, m, real(ylm_mean),
          //        imag(ylm_mean));

          // l=0 should integrate to exactly 1, others should integrate to 0
          if (l == 0) ylm_mean -= 1.0;
          assert(std::abs(ylm_mean) < error_tol);
        }
      }

      // write the orbits to a data file
      FILE* temp_file = fopen("temp.dat", "w");
      lattice.WriteOrbits(temp_file);
      fclose(temp_file);

      // create a new lattice and read the orbit data
      temp_file = fopen("temp.dat", "r");
      QfeLatticeS2 check_lattice(q, k);
      check_lattice.ReadOrbits(temp_file);
      fclose(temp_file);
      remove("temp.dat");

      // check that the site positions match
      for (int s = 0; s < lattice.n_sites; s++) {
        assert(AlmostEq(lattice.r[s], check_lattice.r[s]));
      }
    }
  }
  return 0;
}
