// s2.h

#pragma once

#include <algorithm>
#include <complex>
#include <vector>
#include <map>
#include <string>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <Eigen/Dense>
#include "lattice.h"

typedef std::complex<double> Complex;

class QfeLatticeS2 : public QfeLattice {

public:

  QfeLatticeS2(int q = 5);
  void ResizeSites(int n_sites, int n_dummy = 0);
  void LoopRefine(int n_loop);
  void InterpolateSite(int s, int s_a, int s_b, int num, int den);
  void Inflate();
  void UpdateAntipodes();
  double EdgeSquared(int l);
  double EdgeLength(int l);
  double FlatArea(int f);
  void UpdateWeights();
  void UpdateYlm(int l_max);
  Complex GetYlm(int s, int l, int m);
  double CosTheta(int s1, int s2);
  void PrintCoordinates();

  // site coordinates
  std::vector<Eigen::Vector3d> r;
  std::vector<std::vector<Complex>> ylm;  // spherical harmonics
  std::vector<int> antipode;  // antipode of each site (0 by default)
};

/**
 * @brief Create an unrefined discretization of S2 with @p q links meeting
 * at each site. Valid values for @p q are 3, 4, and 5 for
 * a tetrahedron, octahedron, and icosahedron, respectively.
 */

QfeLatticeS2::QfeLatticeS2(int q) {

  assert(q >= 3 && q <= 5);

  if (q == 3) {

    // tetrahedron
    const double A0 = 0.577350269189625764509148780502L;  // 1/sqrt(3)

    ResizeSites(4);
    for (int s = 0; s < n_sites; s++) {
      sites[s].nn = 0;
      sites[s].wt = 1.0;
      sites[s].id = 0;
    }

    // set coordinates
    r[0] = Eigen::Vector3d( A0,  A0,  A0);
    r[1] = Eigen::Vector3d(-A0, -A0,  A0);
    r[2] = Eigen::Vector3d( A0, -A0, -A0);
    r[3] = Eigen::Vector3d(-A0,  A0, -A0);

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
      sites[s].id = 0;
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
      sites[s].id = 0;
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
    fprintf(stderr, "S2 with q = %d not implemented\n", q);
  }
  n_distinct = 1;
}

/**
 * @brief Change the number of sites.
 */

void QfeLatticeS2::ResizeSites(int n_sites, int n_dummy) {
  QfeLattice::ResizeSites(n_sites, n_dummy);
  r.resize(n_sites + n_dummy);
  ylm.resize(n_sites + n_dummy);
  antipode.resize(n_sites + n_dummy, 0);
}

/**
 * @brief Refine all triangles on the lattice according to the procedure
 * defined in [1]. Each triangle in the original lattice will be split into
 * 2^(@p n_loop) new triangles. This procedure does not project the
 * new sites onto a unit sphere, though it does produce a mesh which is
 * closer to a unit sphere than flat refinement.
 *
 * [1] C. Loop, Smooth Subdivision Surfaces based on Triangles, 1987
 */

void QfeLatticeS2::LoopRefine(int n_loop) {

  const double loop_alpha[] = {
    1.0,
    0.765625,
    0.390625,
    0.4375,
    0.515625,
    0.57953390537108553502L,
    0.625,
    0.65682555866237771184L,
    0.67945752147247766083L,
    0.69593483863689995500L
  };

  const double loop_beta[] = {
    1.0,
    0.61538461538461538462L,
    0.38095238095238095238L,
    0.4,
    0.43636363636363636364L,
    0.47142172687440284144L,
    0.5,
    0.52215726210132291576L,
    0.53914751661736415310L,
    0.55222977312995349577L
  };

  // start with loop's alpha coefficients
  const double* beta_nn = loop_alpha;

  // keep track of the distinct id between any two distinct sites
  std::map<std::string, int> distinct_map;

  for (int i = 0; i <= n_loop; i++) {

    // switch to loop's beta coefficients for the last step
    if (i == n_loop) beta_nn = loop_beta;

    // set coordinates of old sites
    std::vector<Eigen::Vector3d> old_r = r;
    for (int s = 0; s < n_sites; s++) {
      Eigen::Vector3d r_sum = Eigen::Vector3d::Zero();
      int nn = sites[s].nn;
      for (int n = 0; n < nn; n++) {
        r_sum += old_r[sites[s].neighbors[n]];
      }

      double beta = beta_nn[nn];
      r[s] = beta * old_r[s] + r_sum * (1.0 - beta) / double(nn);
    }

    if (i == n_loop) break;

    // copy the old links and faces
    std::vector<QfeLink> old_links = links;
    std::vector<QfeFace> old_faces = faces;

    // remove all links and faces
    links.clear();
    n_links = 0;
    faces.clear();
    n_faces = 0;

    // create new sites
    int n_old_sites = n_sites;
    ResizeSites(n_sites + old_links.size());

    // set coordinates for edge sites
    for (int l = 0; l < old_links.size(); l++) {

      // get the two connected sites
      int s1 = old_links[l].sites[0];
      int s2 = old_links[l].sites[1];
      int s3 = s1;
      int s4 = s2;

      // find the other two adjacent sites
      int f1 = old_links[l].faces[0];
      int f2 = old_links[l].faces[1];

      for (int e = 0; e < 3; e++) {
        s3 = old_faces[f1].sites[e];
        if (s3 != s1 && s3 != s2) break;
      }
      for (int e = 0; e < 3; e++) {
        s4 = old_faces[f2].sites[e];
        if (s4 != s1 && s4 != s2) break;
      }

      r[n_old_sites + l] = 0.125 * (3.0 * (old_r[s1] + old_r[s2]) + \
          old_r[s3] + old_r[s4]);

      // generate a key to identify the distinct site id
      char key[50];
      int id1 = std::min(sites[s1].id, sites[s2].id);
      int id2 = std::max(sites[s1].id, sites[s2].id);
      sprintf(key, "%d_%d", id1, id2);

      if (distinct_map.find(key) == distinct_map.end()) {
        // create a new distinct id
        // printf("%s %d\n", key, n_distinct);
        distinct_map[key] = n_distinct;
        n_distinct++;
      }
      sites[n_old_sites + l].id = distinct_map[key];
    }

    // remove old neighbors
    for (int s = 0; s < n_sites; s++) {
      sites[s].nn = 0;
    }

    // create new faces
    for (int f = 0; f < old_faces.size(); f++) {

      // old sites
      int s1 = old_faces[f].sites[0];
      int s2 = old_faces[f].sites[1];
      int s3 = old_faces[f].sites[2];

      // new sites on old edges
      int s4 = old_faces[f].edges[0] + n_old_sites;
      int s5 = old_faces[f].edges[1] + n_old_sites;
      int s6 = old_faces[f].edges[2] + n_old_sites;

      // add 4 new faces with new links
      AddFace(s1, s6, s4);
      AddFace(s2, s4, s5);
      AddFace(s3, s5, s6);
      AddFace(s4, s6, s5);
    }
  }
}

/**
 * @brief Set the position of site @p s by interpolating between sites @p s_a
 * and @p s_b. The parameters @p num and @den define a fraction between 0 and 1
 * that determines how far from site @p s_a to put site @p s, with values of
 * 0 and 1 giving the coordinates of site a and site b, respectively.
 *
 * When refining triangles on S2, it might seem that interpolating along
 * spherical geodesics would give the most uniform tesselation. However, it
 * can be shown that interpolating in this way gives three different results
 * depending on which axis of the triangle is chosen for the interpolation.
 * Therefore, we interpolate along flat lines in the embedding space. This
 * eliminates the ambiguity in choosing an axis of the triangle, but
 * requires that the interpolated sites must be subsequently projected onto
 * the sphere.
 */

void QfeLatticeS2::InterpolateSite(int s, int s_a, int s_b, int num, int den) {

  // interpolate along a flat line in the embedding space
  double k = double(num) / double(den);
  r[s] = r[s_a] * (1.0 - k) + r[s_b] * k;
}

/**
 * @brief Project all site coordinates onto a unit sphere.
 */

void QfeLatticeS2::Inflate() {
  for (int s = 0; s < n_sites; s++) {
    r[s].normalize();
  }
}

/**
 * @brief Identify each site's antipode, i.e. for a site with position r,
 * find the site which has position -r. A lattice with a tetrahedron base
 * (q = 3) does not have an antipode for every site.
 */

void QfeLatticeS2::UpdateAntipodes() {

  std::map<std::string, int> antipode_map;
  for (int s = 0; s < n_sites; s++) {

    // find antipode
    int x_int = int(round(r[s].x() * 1.0e9));
    int y_int = int(round(r[s].y() * 1.0e9));
    int z_int = int(round(r[s].z() * 1.0e9));
    char key[50];  // keys should be about 32 bytes long
    sprintf(key, "%+d,%+d,%+d", x_int, y_int, z_int);
    char anti_key[50];
    sprintf(anti_key, "%+d,%+d,%+d", -x_int, -y_int, -z_int);

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
    // std::map<std::string, int>::iterator it;
    // for (it = antipode_map.begin(); it != antipode_map.end(); it++) {
    //   fprintf(stderr, "%04d %s\n", it->second, it->first.c_str());
    // }
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
 * @brief Update site and link weights based on vertex coordinates. Sort
 * all sites into groups with the same weight.
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
 * @brief Update spherical harmonic values at each site, up to
 * a maximum l eigenvalue of @p l_max
 */

void QfeLatticeS2::UpdateYlm(int l_max) {

  int n_ylm = ((l_max + 1) * (l_max + 2)) / 2;
  using boost::math::spherical_harmonic;

  for (int s = 0; s < n_sites; s++) {
    ylm[s].resize(n_ylm);
    double theta = acos(r[s].z());
    double phi = atan2(r[s].y(), r[s].x());

    for (int i = 0, l = 0, m = 0; i < n_ylm; i++, m++) {
      if (m > l) {
        m = 0;
        l++;
      }
      assert(i < n_ylm);
      ylm[s][i] = spherical_harmonic(l, m, theta, phi);
    }
  }
}

/**
 * @brief Get the @p l, @p m spherical harmonic at site @p s.
 */

Complex QfeLatticeS2::GetYlm(int s, int l, int m) {

  int abs_m = abs(m);
  assert(abs_m <= l);

  int i = (l * (l + 1)) / 2 + abs_m;
  assert(i < ylm[s].size());
  Complex y = ylm[s][i];

  if (m < 0) {
    y = conj(y);
    if (abs_m & 1) {
      y *= -1;
    }
  }

  return y;
}

/**
 * @brief Return cosine of the angle between sites @p s1 and @p s2. This
 * function assumes that the coordinates have been projected onto the
 * unit sphere.
 */

double QfeLatticeS2::CosTheta(int s1, int s2) {
  return r[s1].dot(r[s2]);
}

/**
 * @brief Print the cartesian coordinates of the sites. This is helpful
 * for making plots in e.g. Mathematica.
 */

void QfeLatticeS2::PrintCoordinates() {
  printf("{");
  for (int s = 0; s < n_sites; s++) {
    printf("{%.12f,%.12f,%.12f}", r[s].x(), r[s].y(), r[s].z());
    printf("%c\n", s == (n_sites - 1) ? '}' : ',');
  }
}
