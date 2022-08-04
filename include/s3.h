// s3.h

#pragma once

#include <cassert>
#include <cmath>
#include <Eigen/Dense>
#include "lattice.h"
#include <boost/math/special_functions/gegenbauer.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

typedef std::complex<double> Complex;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> ComplexMat;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> ComplexVec;

class QfeLatticeS3 : public QfeLattice {

public:

  QfeLatticeS3(int q = 5);
  void ResizeSites(int n_sites);
  void WriteSite(FILE* file, int s);
  void ReadSite(FILE* file, int s);
  void Inflate();
  void UpdateAntipodes();
  Complex GetYjlm(int s, int j, int l, int m);
  Eigen::Vector4d EdgeCenter(int l);
  Eigen::Vector4d FaceCircumcenter(int f);
  Eigen::Vector4d CellCircumcenter(int c);
  double EdgeLength(int l);
  double CellVolume(int c);
  void PrintCoordinates();

  int q;
  std::vector<Eigen::Vector4d> r;
  std::vector<int> antipode;  // antipode of each site (0 by default)
};

QfeLatticeS3::QfeLatticeS3(int q) {

  if (q == 0) return;

  assert(q >= 3 && q <= 5);
  this->q = q;

  if (q == 3) {
    // 5-cell
    const double A0 = 0.0;
    const double A1 = 0.25;
    const double A2 = 0.32274861218395140710L;
    const double A3 = 0.45643546458763842788L;
    const double A4 = 0.79056941504209483300L;
    const double A5 = 0.91287092917527685576L;
    const double A6 = 0.96824583655185422129L;
    const double A7 = 1.0;

    ResizeSites(5);
    for (int s = 0; s < n_sites; s++) {
      sites[s].nn = 0;
      sites[s].wt = 1.0;
      sites[s].id = 0;
    }

    // set coordinates
    r[0] = Eigen::Vector4d(+A7, +A0, +A0, +A0);
    r[1] = Eigen::Vector4d(-A1, +A6, +A0, +A0);
    r[2] = Eigen::Vector4d(-A1, -A2, +A5, +A0);
    r[3] = Eigen::Vector4d(-A1, -A2, -A3, +A4);
    r[4] = Eigen::Vector4d(-A1, -A2, -A3, -A4);

    // add cells (5)
    faces.clear();
    links.clear();
    cells.clear();
    AddCell(0, 1, 2, 3);
    AddCell(0, 1, 2, 4);
    AddCell(0, 1, 3, 4);
    AddCell(0, 2, 3, 4);
    AddCell(1, 2, 3, 4);

  } else if (q == 4) {
    // 16-cell
    ResizeSites(8);
    for (int s = 0; s < n_sites; s++) {
      sites[s].nn = 0;
      sites[s].wt = 1.0;
      sites[s].id = 0;
    }

    // set coordinates
    r[0] = Eigen::Vector4d(+1.0, +0.0, +0.0, +0.0);
    r[1] = Eigen::Vector4d(+0.0, +1.0, +0.0, +0.0);
    r[2] = Eigen::Vector4d(+0.0, +0.0, +1.0, +0.0);
    r[3] = Eigen::Vector4d(+0.0, +0.0, +0.0, +1.0);
    r[4] = Eigen::Vector4d(+0.0, +0.0, +0.0, -1.0);
    r[5] = Eigen::Vector4d(+0.0, +0.0, -1.0, +0.0);
    r[6] = Eigen::Vector4d(+0.0, -1.0, +0.0, +0.0);
    r[7] = Eigen::Vector4d(-1.0, +0.0, +0.0, +0.0);

    // add cells (16)
    faces.clear();
    links.clear();
    cells.clear();
    AddCell(0, 1, 2, 3);
    AddCell(0, 1, 2, 4);
    AddCell(0, 2, 3, 6);
    AddCell(0, 2, 4, 6);
    AddCell(0, 1, 3, 5);
    AddCell(0, 3, 5, 6);
    AddCell(0, 1, 4, 5);
    AddCell(0, 4, 5, 6);
    AddCell(1, 2, 4, 7);
    AddCell(1, 2, 3, 7);
    AddCell(1, 3, 5, 7);
    AddCell(1, 4, 5, 7);
    AddCell(2, 3, 6, 7);
    AddCell(2, 4, 6, 7);
    AddCell(3, 5, 6, 7);
    AddCell(4, 5, 6, 7);

  } else if (q == 5) {
    // 600-cell

  }
}

void QfeLatticeS3::ResizeSites(int n_sites) {
  QfeLattice::ResizeSites(n_sites);
  r.resize(n_sites);
  antipode.resize(n_sites, 0);
}

void QfeLatticeS3::WriteSite(FILE* file, int s) {
  QfeLattice::WriteSite(file, s);
  double xi = acos(r[s].w());
  double theta = 0.0;
  if (xi != 0.0) {
    double cos_theta = r[s].z() / sin(xi);
    if (cos_theta < -1.0) cos_theta = -1.0;
    if (cos_theta > 1.0) cos_theta = 1.0;
    theta = acos(cos_theta);
  }
  double phi = atan2(r[s].y(), r[s].x());
  fprintf(file, " %+.20f %+.20f %+.20f", xi, theta, phi);
}

void QfeLatticeS3::ReadSite(FILE* file, int s) {
  QfeLattice::ReadSite(file, s);
  double xi, theta, phi;
  fscanf(file, " %lf %lf %lf", &xi, &theta, &phi);

  r[s][0] = sin(xi) * sin(theta) * cos(phi);
  r[s][1] = sin(xi) * sin(theta) * sin(phi);
  r[s][2] = sin(xi) * cos(theta);
  r[s][3] = cos(xi);
  r[s].normalize();
}

/**
 * @brief Project all site coordinates onto a unit sphere.
 */

void QfeLatticeS3::Inflate() {
  for (int s = 0; s < n_sites; s++) {
    r[s].normalize();
  }
}

/**
 * @brief Identify each site's antipode, i.e. for a site with position r,
 * find the site which has position -r. A lattice with a 5-cell base
 * (q = 3) does not have an antipode for every site.
 */

void QfeLatticeS3::UpdateAntipodes() {

  std::map<std::string, int> antipode_map;
  char key[100];  // keys should be about 45 bytes long
  char anti_key[100];
  for (int s = 0; s < n_sites; s++) {

    // find antipode
    int x_int = int(round(r[s].x() * 1.0e9));
    int y_int = int(round(r[s].y() * 1.0e9));
    int z_int = int(round(r[s].z() * 1.0e9));
    int w_int = int(round(r[s].w() * 1.0e9));
    sprintf(key, "%+d,%+d,%+d,%+d", x_int, y_int, z_int, w_int);
    sprintf(anti_key, "%+d,%+d,%+d,%+d", -x_int, -y_int, -z_int, -w_int);

    if (antipode_map.find(anti_key) != antipode_map.end()) {
      // antipode found in map
      int a = antipode_map[anti_key];
      antipode[s] = a;
      antipode[a] = s;
      antipode_map.erase(anti_key);
    } else {
      // antipode not found yet
      antipode_map[key] = s;
    }
  }

  if (antipode_map.size()) {
    // print error message if there are any unpaired sites
    fprintf(stderr, "no antipode found for %lu/%d sites\n", \
      antipode_map.size(), n_sites);
  }
}

Complex QfeLatticeS3::GetYjlm(int s, int j, int l, int m) {
  double cos_xi = r[s].w();
  double xi = acos(cos_xi);
  double rho = sin(xi);
  double theta = 0.0;
  if (xi != 0.0) {
    double cos_theta = r[s].z() / rho;
    if (cos_theta < -1.0) cos_theta = -1.0;
    if (cos_theta > 1.0) cos_theta = 1.0;
    theta = acos(cos_theta);
  }
  double phi = atan2(r[s].y(), r[s].x());

  double l_real = double(l);
  double j_real = double(j);
  double c0 = pow(2.0, l_real);
  double c1 = tgamma(l_real + 1.0);
  double c2 = tgamma(j_real - l_real + 1.0);
  double c3 = tgamma(j_real + l_real + 2.0);
  double c4 = 0.79788456080286535588;  // sqrt(2/pi)
  double c5 = j_real + 1.0;
  double C = c0 * c4 * c1 * sqrt(c5 * c2 / c3);
  double S = pow(rho, l_real);
  double G = boost::math::gegenbauer(j - l, l_real + 1.0, cos_xi);
  Complex Y = boost::math::spherical_harmonic(l, m, theta, phi);
  return C * S * G * Y;
}

Eigen::Vector4d QfeLatticeS3::EdgeCenter(int l) {
  Eigen::Vector4d r_a = r[links[l].sites[0]];
  Eigen::Vector4d r_b = r[links[l].sites[1]];

  return 0.5 * (r_a + r_b);
}

Eigen::Vector4d QfeLatticeS3::FaceCircumcenter(int f) {
  Eigen::Vector4d r_a = r[faces[f].sites[0]];
  Eigen::Vector4d r_b = r[faces[f].sites[1]];
  Eigen::Vector4d r_c = r[faces[f].sites[2]];

  Eigen::Vector4d v_ac = r_c - r_a;
  Eigen::Vector4d v_bc = r_c - r_b;

  double A00 = v_ac.dot(v_ac);
  double A01 = v_ac.dot(v_bc);
  double A11 = v_bc.dot(v_bc);

  Eigen::Matrix2d A;
  A(0,0) = A00;
  A(0,1) = A01;
  A(1,0) = A01;
  A(1,1) = A11;

  Eigen::Vector2d b;
  b(0) = A00;
  b(1) = A11;

  Eigen::Vector2d x = 0.5 * A.inverse() * b;

  return x(0) * r_a + x(1) * r_b + (1.0 - x.sum()) * r_c;
}

Eigen::Vector4d QfeLatticeS3::CellCircumcenter(int c) {
  Eigen::Vector4d r_a = r[cells[c].sites[0]];
  Eigen::Vector4d r_b = r[cells[c].sites[1]];
  Eigen::Vector4d r_c = r[cells[c].sites[2]];
  Eigen::Vector4d r_d = r[cells[c].sites[3]];

  Eigen::Vector4d v_ad = r_d - r_a;
  Eigen::Vector4d v_bd = r_d - r_b;
  Eigen::Vector4d v_cd = r_d - r_c;

  double A00 = v_ad.dot(v_ad);
  double A01 = v_ad.dot(v_bd);
  double A02 = v_ad.dot(v_cd);
  double A11 = v_bd.dot(v_bd);
  double A12 = v_bd.dot(v_cd);
  double A22 = v_cd.dot(v_cd);

  Eigen::Matrix3d A;
  A(0,0) = A00;
  A(0,1) = A01;
  A(0,2) = A02;
  A(1,0) = A01;
  A(1,1) = A11;
  A(1,2) = A12;
  A(2,0) = A02;
  A(2,1) = A12;
  A(2,2) = A22;

  Eigen::Vector3d b;
  b(0) = A00;
  b(1) = A11;
  b(2) = A22;

  Eigen::Vector3d x = 0.5 * A.inverse() * b;

  return x(0) * r_a + x(1) * r_b + x(2) * r_c + (1.0 - x.sum()) * r_d;
}

double QfeLatticeS3::EdgeLength(int l) {
  int s_a = links[l].sites[0];
  int s_b = links[l].sites[1];
  Eigen::Vector4d dr = r[s_a] - r[s_b];
  return dr.norm();
}

double QfeLatticeS3::CellVolume(int c) {
  Eigen::Vector4d r_a = r[cells[c].sites[0]];
  Eigen::Vector4d r_b = r[cells[c].sites[1]];
  Eigen::Vector4d r_c = r[cells[c].sites[2]];
  Eigen::Vector4d r_d = r[cells[c].sites[3]];

  Eigen::Vector4d v_ad = r_d - r_a;
  Eigen::Vector4d v_bd = r_d - r_b;
  Eigen::Vector4d v_cd = r_d - r_c;

  double A00 = v_ad.dot(v_ad);
  double A01 = v_ad.dot(v_bd);
  double A02 = v_ad.dot(v_cd);
  double A11 = v_bd.dot(v_bd);
  double A12 = v_bd.dot(v_cd);
  double A22 = v_cd.dot(v_cd);

  Eigen::Matrix3d A;
  A(0,0) = A00;
  A(0,1) = A01;
  A(0,2) = A02;
  A(1,0) = A01;
  A(1,1) = A11;
  A(1,2) = A12;
  A(2,0) = A02;
  A(2,1) = A12;
  A(2,2) = A22;

  return sqrt(A.determinant()) / 6.0;
}

/**
 * @brief Print the cartesian coordinates of the sites. This is helpful
 * for making plots in e.g. Mathematica.
 */

void QfeLatticeS3::PrintCoordinates() {
  printf("{");
  for (int s = 0; s < n_sites; s++) {
    printf("{%.12f,%.12f,%.12f,%.12f}", r[s].w(), r[s].x(), r[s].y(), r[s].z());
    printf("%c\n", s == (n_sites - 1) ? '}' : ',');
  }
}
