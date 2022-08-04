// one_cell_lap.cc

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// calculate the DEC Laplacian spectrum for a single tetrahedron

typedef double Real;
typedef Eigen::Vector3<Real> Vec3;

std::mt19937 gen;

// generate a random number between 0 and 1
Real Rand() {
  static std::uniform_real_distribution<Real> dist(0.0, 1.0);
  return dist(gen);
}

void CalcTetrahedron(Real a2, Real a3, Real a4, Real b3, Real b4) {

  // generate a random tetrahedron with unit circumradius
  Real cos_theta2 = 2.0 * a2 - 1.0;
  Real sin_theta2 = sqrt(1.0 - cos_theta2 * cos_theta2);
  Real cos_theta3 = 2.0 * a3 - 1.0;
  Real sin_theta3 = sqrt(1.0 - cos_theta3 * cos_theta3);
  Real cos_theta4 = 2.0 * a4 - 1.0;
  Real sin_theta4 = sqrt(1.0 - cos_theta4 * cos_theta4);

  Real phi3 = 2.0 * M_PI * b3;
  Real phi4 = 2.0 * M_PI * b4;

  // coordinates of circumcenter and vertices
  Vec3 r[5];
  r[0] = Vec3(0.0, 0.0, 0.0);
  r[1] = Vec3(0.0, 0.0, 1.0);
  r[2] = Vec3(sin_theta2, 0.0, cos_theta2);
  r[3] = Vec3(sin_theta3 * cos(phi3), sin_theta3 * sin(phi3), cos_theta3);
  r[4] = Vec3(sin_theta4 * cos(phi4), sin_theta4 * sin(phi4), cos_theta4);

  // // print vertex coordinates
  // printf("{{%.12f, %.12f, %.12f},\n", r[1].x(), r[1].y(), r[1].z());
  // printf("{%.12f, %.12f, %.12f},\n", r[2].x(), r[2].y(), r[2].z());
  // printf("{%.12f, %.12f, %.12f},\n", r[3].x(), r[3].y(), r[3].z());
  // printf("{%.12f, %.12f, %.12f}};\n", r[4].x(), r[4].y(), r[4].z());

  // generate the Cayley-Menger matrix
  Eigen::Matrix<double, 5, 5> CM;
  Eigen::Vector<double, 5> dsq;
  for (int i = 0; i < 5; i++) {
    CM(i,i) = 0.0;
    for (int j = i + 1; j < 5; j++) {
      CM(i,j) = (r[i] - r[j]).squaredNorm();
      CM(j,i) = CM(i,j);
    }
  }

  Eigen::Vector<Real, 5> cell_lhs(1.0, 0.0, 0.0, 0.0, 0.0);
  Eigen::Vector<Real, 5> cell_xi = CM.inverse() * cell_lhs;
  Real cell_cr_sq = -cell_xi(0) / 2.0;

  // check that circumradius is 1
  assert(fabs(cell_cr_sq - 1.0) < 1.0e-10);

  // add up the 12 contributions to the DEC Laplacian
  Eigen::Matrix<Real, 4, 4> lap = Eigen::Matrix<Real, 4, 4>::Zero();
  for (int i = 1; i <= 4; i++) {
    for (int j = i + 1; j <= 4; j++) {

      // find the other two corners
      int k = 1;
      while (k == i || k == j) k++;
      int l = k + 1;
      while (l == i || l == j) l++;

      Real x_ijk = 2.0 * (CM(i,j) * CM(i,k) + CM(i,j) * CM(j,k) + CM(i,k) * CM(j,k)) \
          - (CM(i,j) * CM(i,j) + CM(i,k) * CM(i,k) + CM(j,k) * CM(j,k));
      Real x_ijl = 2.0 * (CM(i,j) * CM(i,l) + CM(i,j) * CM(j,l) + CM(i,l) * CM(j,l)) \
          - (CM(i,j) * CM(i,j) + CM(i,l) * CM(i,l) + CM(j,l) * CM(j,l));

      Real A_tri_ijk = 0.25 * sqrt(x_ijk);
      Real A_tri_ijl = 0.25 * sqrt(x_ijl);
      Real dual_ijk = CM(i,k) + CM(j,k) - CM(i,j);
      Real dual_ijl = CM(i,l) + CM(j,l) - CM(i,j);

      Real h_ijk = sqrt(cell_cr_sq - CM(i,j) * CM(i,k) * CM(j,k) / x_ijk);
      Real h_ijl = sqrt(cell_cr_sq - CM(i,j) * CM(i,l) * CM(j,l) / x_ijl);

      // sign factor
      if (cell_xi(l) < 0.0) h_ijk *= -1.0;
      if (cell_xi(k) < 0.0) h_ijl *= -1.0;

      Real wt = (dual_ijk * h_ijk / A_tri_ijk + dual_ijl * h_ijl / A_tri_ijl) / 16.0;

      // set off-diagonal terms
      lap(i-1,j-1) = -wt;
      lap(j-1,i-1) = -wt;

      // add to diagonal terms
      lap(i-1,i-1) += wt;
      lap(j-1,j-1) += wt;
    }
  }

  // create the solver and analyze M
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, 4, 4>> es;
  es.compute(lap);
  assert(es.info() == Eigen::Success);
  Eigen::VectorXd eig = es.eigenvalues();

  // print eigenvalues
  printf("%+.12f ", eig(0));
  printf("%+.12f ", eig(1));
  printf("%+.12f ", eig(2));
  printf("%+.12f\n", eig(3));
}

void CalcRightTetrahedron() {
  Real a = 2.0 / 3.0;
  Real b = 1.0 / 3.0;
  CalcTetrahedron(a, a, a, b, -b);
}

void CalcRegularTetrahedron() {
  Real a = 1.0 / 3.0;
  Real b = 1.0 / 3.0;
  CalcTetrahedron(a, a, a, b, -b);
}

void CalcRandomTetrahedron() {
  CalcTetrahedron(Rand(), Rand(), Rand(), Rand(), Rand());
}

int main(int argc, char* argv[]) {

  // seed the rng
  unsigned int seed = 1234;
  gen = std::mt19937(seed);

  CalcRegularTetrahedron();
  CalcRightTetrahedron();
  for (int i = 1; i <= 999; i++) {
    CalcRandomTetrahedron();
  }

  return 0;
}
