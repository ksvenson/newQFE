// s2_lap_eigen.cc

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <cassert>
#include <cstdio>
#include <string>

#include "s2.h"
#include "timer.h"

// calculate the eigenvalues of the Laplacian in the diagonal basis

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

  std::vector<Eigen::Triplet<double>> M_elements;

  // add nearest-neighbor interaction terms
  for (int l = 0; l < lattice.n_links; l++) {
    QfeLink* link = &lattice.links[l];
    int a = link->sites[0];
    int b = link->sites[1];
    double K_link = -link->wt / sqrt(lattice.sites[a].wt * lattice.sites[b].wt);
    M_elements.push_back(Eigen::Triplet<double>(a, b, K_link));
    M_elements.push_back(Eigen::Triplet<double>(b, a, K_link));
  }

  // add self-interaction terms
  for (int s = 0; s < lattice.n_sites; s++) {
    QfeSite* site = &lattice.sites[s];
    double wt_sum = 0.0;
    for (int n = 0; n < site->nn; n++) {
      int l = site->links[n];
      wt_sum += lattice.links[l].wt;
    }
    wt_sum /= lattice.sites[s].wt;
    M_elements.push_back(Eigen::Triplet<double>(s, s, wt_sum));
  }

  Eigen::SparseMatrix<double> M(lattice.n_sites, lattice.n_sites);
  M.setFromTriplets(M_elements.begin(), M_elements.end());

  Timer solve_timer;

#if 0  // use Eigen solver (full eigenspectrum)

  // create the solver and analyze M
  Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> es;
  es.compute(M);
  assert(es.info() == Eigen::Success);
  Eigen::VectorXd eig = es.eigenvalues() / (a_lat * a_lat);

#else  // use Spectra solver (partial eigenspectrum)

  // Construct matrix operation object using the wrapper class DenseSymMatProd
  Spectra::SparseSymMatProd<double> op(M);

  // Construct eigen solver object, requesting the smallest 169 eigenvalues (up
  // to l=12) total number of eigenvalues up to l is (l + 1)^2
  int ncv = 180 * 2;
  if (lattice.n_sites < ncv) ncv = lattice.n_sites;
  int nev = 180;
  if (nev > (ncv - 2)) nev = ncv - 2;
  printf("nev: %d\n", nev);
  printf("ncv: %d\n", ncv);
  Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> es(op, nev, ncv);

  // Initialize and compute
  es.init();
  int nconv = es.compute(Spectra::SortRule::SmallestAlge);
  printf("nconv: %d\n", nconv);

  // Retrieve results
  assert(es.info() == Spectra::CompInfo::Successful);
  Eigen::VectorXd eig = es.eigenvalues() / (a_lat * a_lat);
  eig.reverseInPlace();

#endif

  solve_timer.Stop();
  printf("solve time: %.12f\n", solve_timer.Duration());

  for (int i = 0; i < eig.size(); i++) {
    printf("%04d %.12f\n", i, Chop(eig(i)));
  }

  return 0;
}
