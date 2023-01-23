// s3_test.cc

#include "s3.h"

#include <cstdio>

int main(int argc, char* argv[]) {
  // list of refinement values to test
  int k_list[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 28, 32};
  int n_k = sizeof(k_list) / sizeof(int);

  for (int q = 3; q <= 5; q++) {
    for (int i_k = 0; i_k < n_k; i_k++) {
      int k = k_list[i_k];
      printf("Testing q%dk%d lattice\n", q, k);
      QfeLatticeS3 lattice(q, k);

      // check Euler characteristic
      int euler_chi =
          lattice.n_sites - lattice.n_links + lattice.n_faces - lattice.n_cells;
      // printf("euler characteristic: %d\n", euler_chi);
      assert(euler_chi == 0);

      // check antipodal points
      if (q != 3) lattice.UpdateAntipodes();

      // calculate DEC weights
      lattice.CalcFEMWeights();

      // check that low ell irreps of O(4) can be integrated exactly
      double error_tol = 1.0e-9;
      int j_max;
      if (q == 3) {
        j_max = 2;
      } else if (q == 4) {
        j_max = 3;
      } else {
        j_max = 5;
      }
      for (int j = 0; j <= j_max; j++) {
        for (int l = 0; l <= j; l++) {
          for (int m = -l; m <= l; m++) {
            Complex yjlm_sum = 0.0;
            for (int s = 0; s < lattice.n_sites; s++) {
              yjlm_sum += lattice.sites[s].wt * lattice.GetYjlm(s, j, l, m);
            }
            Complex yjlm_mean =
                yjlm_sum * sqrt(2.0 * M_PI * M_PI) / lattice.vol;
            // printf("j,l,m=%d,%d,%+d: %+.12e %+.12e\n", j, l, m,
            // real(yjlm_mean),
            //        imag(yjlm_mean));

            // j=0 should integrate to exactly 1, others should integrate to 0
            if (j == 0) yjlm_mean -= 1.0;
            assert(std::abs(yjlm_mean) < error_tol);
          }
        }
      }

      // write the orbits to a data file
      FILE* temp_file = fopen("temp.dat", "w");
      lattice.WriteOrbits(temp_file);
      fclose(temp_file);

      // create a new lattice and read the orbit data
      temp_file = fopen("temp.dat", "r");
      QfeLatticeS3 check_lattice(q, k);
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
