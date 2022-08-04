// s3_lap_eigen.cc

#include <cassert>
#include <cstdio>
#include <string>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include "s3.h"
#include "timer.h"

// calculate the eigenvalues of the Laplacian in the diagonal basis

int main(int argc, char* argv[]) {

  Timer solve_timer;

  char base_path[200];
  sprintf(base_path, "%s", "s3_riesz/q5v1");
  if (argc > 1) {
    sprintf(base_path, "%s", argv[1]);
  }

  // read site coordinates
  QfeLatticeS3 lattice(0);
  char lattice_path[200];
  sprintf(lattice_path, "%s_lattice.dat", base_path);
  FILE* lattice_file = fopen(lattice_path, "r");
  assert(lattice_file != nullptr);
  lattice.ReadLattice(lattice_file);
  fclose(lattice_file);

  std::vector<Eigen::Triplet<double>> M_elements;

  // add nearest-neighbor interaction terms
  for (int l = 0; l < lattice.n_links; l++) {
    QfeLink* link = &lattice.links[l];
    int a = link->sites[0];
    int b = link->sites[1];
    M_elements.push_back(Eigen::Triplet<double>(a, b, -link->wt));
    M_elements.push_back(Eigen::Triplet<double>(b, a, -link->wt));
  }

  // add self-interaction terms
  for (int s = 0; s < lattice.n_sites; s++) {
    QfeSite* site = &lattice.sites[s];
    double wt_sum = 0.0;
    for (int n = 0; n < site->nn; n++) {
      int l = site->links[n];
      wt_sum += lattice.links[l].wt;
    }
    M_elements.push_back(Eigen::Triplet<double>(s, s, wt_sum));
  }

  Eigen::SparseMatrix<double> M(lattice.n_sites, lattice.n_sites);
  M.setFromTriplets(M_elements.begin(), M_elements.end());

#if 0  // use Eigen solver (full eigenspectrum)

  // create the solver and analyze M
  Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> es;
  es.compute(M);
  assert(es.info() == Eigen::Success);
  Eigen::VectorXd eig = es.eigenvalues() / (2.0 * M_PI * M_PI) * double(lattice.n_sites);

#else  // use Spectra solver (partial eigenspectrum)

  // Construct matrix operation object using the wrapper class DenseSymMatProd
  Spectra::SparseSymMatProd<double> op(M);

  // Construct eigen solver object, requesting the smallest 819 eigenvalues (up to j=12)
  int ncv = 850 * 2;  // was 819
  if (lattice.n_sites < ncv) ncv = lattice.n_sites;
  int nev = 850;
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
  Eigen::VectorXd eig = es.eigenvalues() * double(lattice.n_sites) / (2.0 * M_PI * M_PI);
  eig.reverseInPlace();

#endif

  solve_timer.Stop();
  printf("solve time: %.12f\n", solve_timer.Duration());

  char eigen_path[200];
  sprintf(eigen_path, "%s_eigen.dat", base_path);
  FILE* eigen_file = fopen(eigen_path, "w");
  assert(eigen_file != nullptr);
  for (int i = 0; i < eig.size(); i++) {
    // printf("%06d %.12f\n", i, eig(i));
    fprintf(eigen_file, "%04d %.12f\n", i, eig(i));
  }
  fclose(eigen_file);

  return 0;
}
