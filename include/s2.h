// s2.h

#pragma once

#include <vector>
#include <Eigen/Dense>
#include "lattice.h"

class QfeLatticeS2 : public QfeLattice {

public:

  QfeLatticeS2(int q = 5);
  void ResizeSites(int n_sites, int n_dummy = 0);
  void InterpolateSite(int s, int s_a, int s_b, double k);
  void Inflate();
  double EdgeSquared(int l);
  double EdgeLength(int l);
  double FlatArea(int f);
  void UpdateWeights();
  void PrintCoordinates();

  // site coordinates
  std::vector<Eigen::Vector3d> r;
  std::vector<double> theta;
  std::vector<double> phi;
};

/**
 * @brief Create an unrefined discretization of S2 with @p q links meeting
 * at each site. Valid values for @p q are 3, 4, and 5 for
 * a tetrahedron, octahedron, and icosahedron, respectively.
 */

QfeLatticeS2::QfeLatticeS2(int q) {

  if (q == 3) {

    // tetrahedron
    const double A0 = 0.0L;
    const double A1 = 0.333333333333333333333333333333L;
    const double A2 = 0.471404520791031682933896241403L;
    const double A3 = 0.816496580927726032732428024902L;
    const double A4 = 0.942809041582063365867792482806L;
    const double A5 = 1.0L;

    ResizeSites(4);
    for (int s = 0; s < n_sites; s++) {
      sites[s].nn = 0;
      sites[s].wt = 1.0;
    }

    // set coordinates
    r[0] = Eigen::Vector3d( A0,  A0,  A5);
    r[1] = Eigen::Vector3d( A4,  A0, -A1);
    r[2] = Eigen::Vector3d(-A2,  A3, -A1);
    r[3] = Eigen::Vector3d(-A2, -A3, -A1);

    // add faces (4)
    faces.clear();
    links.clear();
    AddFace(0, 1, 2);
    AddFace(0, 2, 3);
    AddFace(0, 3, 1);
    AddFace(1, 3, 2);

  } else if (q == 4) {

    // octahedron
    ResizeSites(6);
    for (int s = 0; s < n_sites; s++) {
      sites[s].nn = 0;
      sites[s].wt = 1.0;
    }

    // set coordinates
    r[0] = Eigen::Vector3d( 0.0,  0.0,  1.0);
    r[1] = Eigen::Vector3d( 1.0,  0.0,  0.0);
    r[2] = Eigen::Vector3d( 0.0,  1.0,  0.0);
    r[3] = Eigen::Vector3d(-1.0,  0.0,  0.0);
    r[4] = Eigen::Vector3d( 0.0, -1.0,  0.0);
    r[5] = Eigen::Vector3d( 0.0,  0.0, -1.0);

    // add faces (8)
    faces.clear();
    links.clear();
    AddFace(0, 1, 2);
    AddFace(0, 2, 3);
    AddFace(0, 3, 4);
    AddFace(0, 4, 1);
    AddFace(1, 5, 2);
    AddFace(2, 5, 3);
    AddFace(3, 5, 4);
    AddFace(4, 5, 1);

  } else if (q == 5) {

    // icosahedron
    const double C0 = 0.0L;
    const double C1 = 0.276393202250021030359082633127L;
    const double C2 = 0.447213595499957939281834733746L;
    const double C3 = 0.525731112119133606025669084848L;
    const double C4 = 0.723606797749978969640917366873L;
    const double C5 = 0.850650808352039932181540497063L;
    const double C6 = 0.894427190999915878563669467493L;
    const double C7 = 1.0L;

    ResizeSites(12);
    for (int s = 0; s < n_sites; s++) {
      sites[s].nn = 0;
      sites[s].wt = 1.0;
    }

    // set coordinates
    r[0]  = Eigen::Vector3d( C0,  C0,  C7);
    r[1]  = Eigen::Vector3d( C6,  C0,  C2);
    r[2]  = Eigen::Vector3d( C1,  C5,  C2);
    r[3]  = Eigen::Vector3d(-C4,  C3,  C2);
    r[4]  = Eigen::Vector3d(-C4, -C3,  C2);
    r[5]  = Eigen::Vector3d( C1, -C5,  C2);
    r[6]  = Eigen::Vector3d( C4, -C3, -C2);
    r[7]  = Eigen::Vector3d( C4,  C3, -C2);
    r[8]  = Eigen::Vector3d(-C1,  C5, -C2);
    r[9]  = Eigen::Vector3d(-C6,  C0, -C2);
    r[10] = Eigen::Vector3d(-C1, -C5, -C2);
    r[11] = Eigen::Vector3d( C0,  C0, -C7);

    // add faces (20)
    faces.clear();
    links.clear();
    AddFace(0, 1, 2);
    AddFace(0, 2, 3);
    AddFace(0, 3, 4);
    AddFace(0, 4, 5);
    AddFace(0, 5, 1);
    AddFace(1, 6, 7);
    AddFace(1, 7, 2);
    AddFace(2, 7, 8);
    AddFace(2, 8, 3);
    AddFace(3, 8, 9);
    AddFace(3, 9, 4);
    AddFace(4, 9, 10);
    AddFace(4, 10, 5);
    AddFace(5, 10, 6);
    AddFace(5, 6, 1);
    AddFace(6, 11, 7);
    AddFace(7, 11, 8);
    AddFace(8, 11, 9);
    AddFace(9, 11, 10);
    AddFace(10, 11, 6);

  } else {
    printf("S2 with q = %d not implemented\n", q);
  }
}

/**
 * @brief Change the number of sites.
 */

void QfeLatticeS2::ResizeSites(int n_sites, int n_dummy) {
  QfeLattice::ResizeSites(n_sites, n_dummy);
  r.resize(n_sites + n_dummy);
  theta.resize(n_sites + n_dummy);
  phi.resize(n_sites + n_dummy);
}

/**
 * @brief Interpolate coordinates partway between two sites.
 */

void QfeLatticeS2::InterpolateSite(int s, int s_a, int s_b, double k) {

  // // interpolate along a spherical geodesic
  // Eigen::Vector3d p = r[s_a].cross(r[s_b]).normalized();
  // double psi = acos(r[s_a].dot(r[s_b])) * k;
  // r[s] = r[s_a] * cos(psi) + p.cross(r[s_a]) * sin(psi) + \
  //     p * p.dot(r[s_a]) * (1.0 - cos(psi));

  // interpolate along a flat line
  r[s] = r[s_a] * (1.0 - k) + r[s_b] * k;
}

/**
 * @brief Project all site coordinates onto a unit sphere.
 */

void QfeLatticeS2::Inflate() {
  for (int s = 0; s < n_sites; s++) {
    r[s].normalize();
    theta[s] = acos(r[s].z());
    phi[s] = atan2(r[s].y(), r[s].x());
  }
}

/**
 * @brief Calculate the squared length of a link
 */

double QfeLatticeS2::EdgeSquared(int l) {
  int s_a = links[l].sites[0];
  int s_b = links[l].sites[1];
  Eigen::Vector3d dr = r[s_a] - r[s_b];
  return dr.squaredNorm();
}

/**
 * @brief Calculate the length of a link
 */

double QfeLatticeS2::EdgeLength(int l) {
  return sqrt(EdgeSquared(l));
}

/**
 * @brief Calculate the flat area of a triangular face
 */

double QfeLatticeS2::FlatArea(int f) {
  double a = EdgeLength(faces[f].edges[0]);
  double b = EdgeLength(faces[f].edges[1]);
  double c = EdgeLength(faces[f].edges[2]);
  double area = (a + b + c) * (b + c - a) * (c + a - b) * (a + b - c);
  return 0.25 * sqrt(area);
}

/**
 * @brief Update site and link weights based on vertex coordinates.
 */

void QfeLatticeS2::UpdateWeights() {

  // set site weights to zero
  for (int s = 0; s < n_sites; s++) {
    sites[s].wt = 0.0;
  }

  // loop over links to update weights
  double link_wt_sum = 0.0;
  for (int l = 0; l < n_links; l++) {
    links[l].wt = 0.0;
    for (int i = 0; i < 2; i++) {

      // find the other two edges of this face
      int f = links[l].faces[i];
      int e = 0;
      if (faces[f].edges[0] == l) {
        e = 0;
      } else if (faces[f].edges[1] == l) {
        e = 1;
      } else if (faces[f].edges[2] == l) {
        e = 2;
      } else {
        printf("invalid face %04d for link %04d\n", f, l);
      }
      int e1 = (e + 1) % 3;
      int e2 = (e + 2) % 3;
      int l1 = faces[f].edges[e1];
      int l2 = faces[f].edges[e2];

      // find the area associated with this face
      double sq_edge = EdgeSquared(l);
      double sq_edge_1 = EdgeSquared(l1);
      double sq_edge_2 = EdgeSquared(l2);
      double half_wt = 0.25 * (sq_edge_1 + sq_edge_2 - sq_edge) / FlatArea(f);
      links[l].wt += half_wt;

      // add to the weights of the two sites connected by this link
      sites[links[l].sites[0]].wt += half_wt * sq_edge;
      sites[links[l].sites[1]].wt += half_wt * sq_edge;
    }
    link_wt_sum += links[l].wt;
  }
  double link_wt_norm = 1.5 * link_wt_sum / double(n_links);

  // normalize link weights
  for (int l = 0; l < n_links; l++) {
    links[l].wt /= link_wt_norm;
  }

  // normalize site weights to 1
  double site_wt_sum = 0.0;
  for (int s = 0; s < n_sites; s++) {
    site_wt_sum += sites[s].wt;
  }

  double site_wt_norm = site_wt_sum / double(n_sites);

  for (int s = 0; s < n_sites; s++) {
    sites[s].wt /= site_wt_norm;
  }
}

/**
 * @brief Print the cartesian coordinates of the sites. This is helpful
 * for making plots in e.g. Mathematica.
 */

void QfeLatticeS2::PrintCoordinates() {
  for (int s = 0; s < n_sites; s++) {
    printf("{%.12f, %.12f, %.12f},\n", r[s].x(), r[s].y(), r[s].z());
  }
}
