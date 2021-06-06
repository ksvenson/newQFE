// s2.h

#pragma once

#include <vector>
#include "lattice.h"

using std::vector;

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

  int q;  // number of triangles meeting at corner sites

  // site coordinates
  vector<double> x;
  vector<double> y;
  vector<double> z;
  vector<double> cos_theta;
  vector<double> phi;

};

QfeLatticeS2::QfeLatticeS2(int q) {
  this->q = q;

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
    x[0] =  A0; y[0] =  A0; z[0] =  A5;
    x[1] =  A4; y[1] =  A0; z[1] = -A1;
    x[2] = -A2; y[2] =  A3; z[2] = -A1;
    x[3] = -A2; y[3] = -A3; z[3] = -A1;

    // add faces (4)
    faces.clear();
    links.clear();
    AddFace(0, 1, 2);
    AddFace(0, 2, 3);
    AddFace(0, 3, 1);
    AddFace(1, 3, 2);

    // // add links (6)
    // AddLink(0, 1, 1.0);
    // AddLink(0, 2, 1.0);
    // AddLink(0, 3, 1.0);
    // AddLink(1, 2, 1.0);
    // AddLink(2, 3, 1.0);
    // AddLink(3, 1, 1.0);

  } else if (q == 4) {

    // octahedron
    ResizeSites(6);
    for (int s = 0; s < n_sites; s++) {
      sites[s].nn = 0;
      sites[s].wt = 1.0;
    }

    // set coordinates
    x[0] =  0.0; y[0] =  0.0; z[0] =  1.0;
    x[1] =  1.0; y[1] =  0.0; z[1] =  0.0;
    x[2] =  0.0; y[2] =  1.0; z[2] =  0.0;
    x[3] = -1.0; y[3] =  0.0; z[3] =  0.0;
    x[4] =  0.0; y[4] = -1.0; z[4] =  0.0;
    x[5] =  0.0; y[5] =  0.0; z[5] = -1.0;

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

    // // add links (12)
    // links.clear();
    // AddLink(0, 1, 1.0);
    // AddLink(0, 2, 1.0);
    // AddLink(0, 3, 1.0);
    // AddLink(0, 4, 1.0);
    // AddLink(1, 2, 1.0);
    // AddLink(2, 3, 1.0);
    // AddLink(3, 4, 1.0);
    // AddLink(4, 1, 1.0);
    // AddLink(1, 5, 1.0);
    // AddLink(2, 5, 1.0);
    // AddLink(3, 5, 1.0);
    // AddLink(4, 5, 1.0);

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
    x[0]  =  C0; y[0]  =  C0; z[0]  =  C7;
    x[1]  =  C6; y[1]  =  C0; z[1]  =  C2;
    x[2]  =  C1; y[2]  =  C5; z[2]  =  C2;
    x[3]  = -C4; y[3]  =  C3; z[3]  =  C2;
    x[4]  = -C4; y[4]  = -C3; z[4]  =  C2;
    x[5]  =  C1; y[5]  = -C5; z[5]  =  C2;
    x[6]  =  C4; y[6]  = -C3; z[6]  = -C2;
    x[7]  =  C4; y[7]  =  C3; z[7]  = -C2;
    x[8]  = -C1; y[8]  =  C5; z[8]  = -C2;
    x[9]  = -C6; y[9]  =  C0; z[9]  = -C2;
    x[10] = -C1; y[10] = -C5; z[10] = -C2;
    x[11] =  C0; y[11] =  C0; z[11] = -C7;

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

    // // add links (30)
    // links.clear();
    // AddLink(0, 1, 1.0);
    // AddLink(0, 2, 1.0);
    // AddLink(0, 3, 1.0);
    // AddLink(0, 4, 1.0);
    // AddLink(0, 5, 1.0);
    // AddLink(1, 2, 1.0);
    // AddLink(2, 3, 1.0);
    // AddLink(3, 4, 1.0);
    // AddLink(4, 5, 1.0);
    // AddLink(5, 1, 1.0);
    // AddLink(1, 6, 1.0);
    // AddLink(1, 7, 1.0);
    // AddLink(2, 7, 1.0);
    // AddLink(2, 8, 1.0);
    // AddLink(3, 8, 1.0);
    // AddLink(3, 9, 1.0);
    // AddLink(4, 9, 1.0);
    // AddLink(4, 10, 1.0);
    // AddLink(5, 10, 1.0);
    // AddLink(5, 6, 1.0);
    // AddLink(6, 7, 1.0);
    // AddLink(7, 8, 1.0);
    // AddLink(8, 9, 1.0);
    // AddLink(9, 10, 1.0);
    // AddLink(10, 6, 1.0);
    // AddLink(6, 11, 1.0);
    // AddLink(7, 11, 1.0);
    // AddLink(8, 11, 1.0);
    // AddLink(9, 11, 1.0);
    // AddLink(10, 11, 1.0);

  } else {
    printf("S2 with q = %d not implemented\n", q);
  }
}

/**
 * @brief Change the number of sites.
 */

void QfeLatticeS2::ResizeSites(int n_sites, int n_dummy) {
  QfeLattice::ResizeSites(n_sites, n_dummy);
  x.resize(n_sites + n_dummy);
  y.resize(n_sites + n_dummy);
  z.resize(n_sites + n_dummy);
  cos_theta.resize(n_sites + n_dummy);
  phi.resize(n_sites + n_dummy);
}

void QfeLatticeS2::InterpolateSite(int s, int s_a, int s_b, double k) {
  x[s] = x[s_a] * (1.0 - k) + x[s_b] * k;
  y[s] = y[s_a] * (1.0 - k) + y[s_b] * k;
  z[s] = z[s_a] * (1.0 - k) + z[s_b] * k;
}

/**
 * @brief Project all site coordinates onto a unit sphere.
 */

void QfeLatticeS2::Inflate() {
  for (int s = 0; s < n_sites; s++) {
    double x_s = x[s];
    double y_s = y[s];
    double z_s = z[s];
    double r = sqrt(x_s * x_s + y_s * y_s + z_s * z_s);
    x[s] /= r;
    y[s] /= r;
    z[s] /= r;
    cos_theta[s] = z[s];
    phi[s] = atan2(y[s], x[s]);
  }
}

/**
 * @brief Calculate the squared length of a link
 */

double QfeLatticeS2::EdgeSquared(int l) {
  int s_a = links[l].sites[0];
  int s_b = links[l].sites[1];
  double dx = x[s_a] - x[s_b];
  double dy = y[s_a] - y[s_b];
  double dz = z[s_a] - z[s_b];
  return dx * dx + dy * dy + dz * dz;
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

  // normalize link weights to 1
  double site_wt_sum = 0.0;
  for (int s = 0; s < n_sites; s++) {
    site_wt_sum += sites[s].wt;
  }

  double site_wt_norm = site_wt_sum / double(n_sites);

  for (int s = 0; s < n_sites; s++) {
    sites[s].wt /= site_wt_norm;
  }
}
