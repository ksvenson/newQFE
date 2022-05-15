// s2.h

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <complex>
#include <vector>
#include <map>
#include <string>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include "lattice.h"

typedef std::complex<double> Complex;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> ComplexMat;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> ComplexVec;

class QfeLatticeS2 : public QfeLattice {

public:

  QfeLatticeS2(int q = 5);
  void WriteSite(FILE* file, int s);
  void ReadSite(FILE* file, int s);
  void ResizeSites(int n_sites);
  void LoopRefine(int n_loop);
  void InterpolateSite(int s, int s_a, int s_b, int num, int den);
  void Inflate();
  void UpdateAntipodes();
  Eigen::Vector3d FaceCircumcenter(int f);
  double EdgeSquared(int l);
  double EdgeLength(int l);
  double FlatArea(int f);
  void UpdateWeights();
  void OptimizeIntegrator(int l_max);
  void UpdateYlm(int l_max);
  Complex GetYlm(int s, int l, int m);
  double CosTheta(int s1, int s2);
  void PrintCoordinates();
  void UpdateTriangleCoordinates();

  // site coordinates
  int q;
  std::vector<Eigen::Vector3d> r;
  std::vector<double> tri2;  // s^2 + t^2 + u^2
  std::vector<double> tri3;  // stu
  std::vector<std::vector<Complex>> ylm;  // spherical harmonics
  std::vector<int> antipode;  // antipode of each site (0 by default)
};

/**
 * @brief Create an unrefined discretization of S2 with @p q links meeting
 * at each site. Valid values for @p q are 3, 4, and 5 for
 * a tetrahedron, octahedron, and icosahedron, respectively.
 */

QfeLatticeS2::QfeLatticeS2(int q) {

  // create an empty lattice if q = 0
  if (q == 0) return;

  assert(q >= 3 && q <= 5);
  this->q = q;

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

void QfeLatticeS2::WriteSite(FILE* file, int s) {
  QfeLattice::WriteSite(file, s);
  double theta = acos(r[s].z());
  double phi = atan2(r[s].y(), r[s].x());
  fprintf(file, " %+.20f %+.20f", theta, phi);
}

void QfeLatticeS2::ReadSite(FILE* file, int s) {
  QfeLattice::ReadSite(file, s);
  double theta, phi;
  fscanf(file, " %lf %lf", &theta, &phi);

  r[s][0] = sin(theta) * cos(phi);
  r[s][1] = sin(theta) * sin(phi);
  r[s][2] = cos(theta);
  r[s].normalize();
}

/**
 * @brief Change the number of sites.
 */

void QfeLatticeS2::ResizeSites(int n_sites) {
  QfeLattice::ResizeSites(n_sites);
  r.resize(n_sites);
  ylm.resize(n_sites);
  antipode.resize(n_sites, 0);
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
      AddFace(s1, s5, s6);
      AddFace(s2, s4, s6);
      AddFace(s3, s4, s5);
      AddFace(s4, s5, s6);
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
 * @brief Find the circumcenter of face @p f
 */

Eigen::Vector3d QfeLatticeS2::FaceCircumcenter(int f) {
  double sq_edge_1 = EdgeSquared(faces[f].edges[0]);
  double sq_edge_2 = EdgeSquared(faces[f].edges[1]);
  double sq_edge_3 = EdgeSquared(faces[f].edges[2]);

  double w1 = sq_edge_1 * (sq_edge_2 + sq_edge_3 - sq_edge_1);
  double w2 = sq_edge_2 * (sq_edge_3 + sq_edge_1 - sq_edge_2);
  double w3 = sq_edge_3 * (sq_edge_1 + sq_edge_2 - sq_edge_3);

  Eigen::Vector3d r1 = w1 * r[faces[f].sites[0]];
  Eigen::Vector3d r2 = w2 * r[faces[f].sites[1]];
  Eigen::Vector3d r3 = w3 * r[faces[f].sites[2]];

  return (r1 + r2 + r3) / (w1 + w2 + w3);
}

/**
 * @brief Calculate the squared length of link @p l
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
      while (faces[f].edges[e] != l) e++;
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
  this->vol = double(n_sites);
}

/**
 * @brief Optimize the site weights so that all linear combinations of
 * spherical harmonics invariant under the relevant symmetry group can be
 * integrated exactly up to order @p l_max.
 */

void QfeLatticeS2::OptimizeIntegrator(int l_max) {

  // determine which l,m combinations need to be integrated exactly
  std::vector<int> l_relevant;
  std::vector<int> m_relevant;
  int m_spacing = (q == 5) ? 5 : 4;

  for (int l = 0; l <= l_max; l++) {
    // odd l only contributes for tetrahedron
    if ((l % 2) && (q != 3)) continue;

    // number of functions at this l (overcounts for tetrahedron)
    int n_l = (l / 2) + (l / 3) + (l / q) - l + 1;

    for (int i = 0; i < n_l; i++) {
      l_relevant.push_back(l);
      int m = i * m_spacing;

      // odd ell (tetrahedron only)
      if (l % 2) m = ((2 * i + 1) * m_spacing) / 2;

      m_relevant.push_back(m);
    }
  }

  // number of relevant functions
  int n_relevant = l_relevant.size();

  // generate the rectangular matrix
  ComplexMat S = ComplexMat::Zero(n_relevant, n_distinct);
  for (int s = 0; s < n_sites; s++) {
    int id = sites[s].id;
    double theta = acos(r[s].z());
    double phi = atan2(r[s].y(), r[s].x());
    for (int i = 0; i < n_relevant; i++) {
      int l = l_relevant[i];
      int m = m_relevant[i];
      S(i,id) += boost::math::spherical_harmonic(l, m, theta, phi);
    }
  }

  // use the current weights as an initial guess
  ComplexVec x0(n_distinct);
  for (int id = 0; id < n_distinct; id++) {
    int s = distinct_first[id];
    x0(id) = sites[s].wt;
  }

  // the right hand side is the spherical harmonic orthogonality condition
  ComplexVec b = ComplexVec::Zero(n_relevant);
  b(0) = 0.28209479177387814347 * vol;  // n_sites / sqrt(4 pi)

  // compute the solution
  Eigen::LeastSquaresConjugateGradient<ComplexMat> cg;
  cg.compute(S);
  assert(cg.info() == Eigen::Success);
  ComplexVec wt = cg.solveWithGuess(b, x0);

  // apply the improved weights to the sites
  for (int s = 0; s < n_sites; s++) {
    int id = sites[s].id;
    if (s == distinct_first[id]) {
      assert(real(wt(id)) > 0.0);
      // printf("%04d %.12f %.12f\n", id, sites[s].wt, real(wt(id)));
    }
    sites[s].wt = real(wt(id));
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

  int abs_m = fabs(m);
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
  if (s1 == s2) return 1.0;
  if (antipode[s1] == s2) return -1.0;
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

/**
 * @brief For each distinct group, calculate quantities that are invariant
 * under the symmetries of the triangle group. Values are normalized to 1
 * at the polyhedral vertices. The quadratic and cubic invariants are
 * s^2 + t^2 + u^2 and stu where s, t, and u are the coordinates of the site
 * projected onto the axes of the triangular polyhedral faces. This function
 * should be called after refining the polyhedral faces but *before* calling
 * Inflate().
 */

void QfeLatticeS2::UpdateTriangleCoordinates() {
  UpdateDistinct();
  tri2.resize(n_distinct);
  tri3.resize(n_distinct);

  // center of first polyhedral face
  Eigen::Vector3d c = (r[0] + r[1] + r[2]) / 3.0;

  // vectors along two axes of the first polyhedral face
  Eigen::Vector3d v_s = r[0] - c;
  Eigen::Vector3d v_t = r[1] - c;

  // distance between center and polyhedral vertices
  double norm_v = v_s.squaredNorm();

  for (int d = 0; d < n_distinct; d++) {

    // the first distinct point should be in the first polyhedral face
    int s = distinct_first[d];

    // coordinate of lattice point in polyhedral face
    Eigen::Vector3d x = r[s] - c;

    // project coordinates onto triangular axes
    double u_s = v_s.dot(x) / norm_v;
    double u_t = v_t.dot(x) / norm_v;

    // quadratic triangular invariant s^2 + t^2 + u^2
    tri2[d] = (u_s * u_s + u_s * u_t + u_t * u_t) / 0.75;

    // cubic triangular invariant stu
    tri3[d] = -4.0 * u_s * u_t * (u_s + u_t);
  }
}
