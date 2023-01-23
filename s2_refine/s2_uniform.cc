// s2_std_uniform.cc

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/IterativeLinearSolvers>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <set>
#include <vector>

#include "s2.h"
#include "timer.h"
#include "util.h"

typedef double Real;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> Mat;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> Vec;

enum OrbitType { V, E, F, VE, VF, EF, VEF };

std::vector<OrbitType> orbit_type;

std::vector<int> distinct_n_faces;
std::vector<int> distinct_first_face;
std::set<int> primary_sites;

void UpdateOrbit(QfeLatticeS2& lattice, int o) {
  Timer timer;
  Vec3 xi = lattice.orbit_xi[o];
  OrbitType type = orbit_type[o];
  switch (type) {
    case OrbitType::VE:
      xi[1] = 1.0 - xi[0];
      break;
    case OrbitType::VF:
      xi[1] = 0.5 - 0.5 * xi[0];
      break;
    case OrbitType::EF:
      xi[1] = xi[0];
      break;
    case OrbitType::VEF:
      break;
    default:
      printf("Invalid orbit %d\n", type);
      exit(1);
      break;
  }

  xi[2] = 1.0 - (xi[0] + xi[1]);

  lattice.orbit_xi[o] = xi;

  Vec3 v = lattice.CalcOrbitPos(o);

  // use the symmetry group data to calculate site coordinates
  std::set<int>::iterator it = primary_sites.begin();
  while (it != primary_sites.end()) {
    int s = *it++;
    if (lattice.sites[s].id != o) continue;
    int g = lattice.site_g[s];
    lattice.r[s] = (lattice.G[g] * v).normalized();
  }
}

double FaceAreaError(QfeLatticeS2& lattice) {
  double area_sum = 0.0;
  double area_sq_sum = 0.0;
  int n_total = 0;
  for (int i = 0; i < distinct_first_face.size(); i++) {
    double area = lattice.FlatArea(distinct_first_face[i]);
    area_sum += area * double(distinct_n_faces[i]);
    area_sq_sum += area * area * double(distinct_n_faces[i]);
    n_total += distinct_n_faces[i];
  }
  double area_mean = area_sum / double(n_total);
  double area_sq_mean = area_sq_sum / double(n_total);
  assert(!isnan(area_mean));

  return area_sq_mean / (area_mean * area_mean) - 1.0;
}

double CircumradiusError(QfeLatticeS2& lattice) {
  double area_sum = 0.0;
  double area_sq_sum = 0.0;
  int n_total = 0;
  for (int i = 0; i < distinct_first_face.size(); i++) {
    int f = distinct_first_face[i];
    int s = lattice.faces[f].sites[0];
    Vec3 cc = lattice.FaceCircumcenter(distinct_first_face[i]);
    double cr = (cc - lattice.r[s]).norm();
    double area = cr * cr;
    area_sum += area * double(distinct_n_faces[i]);
    area_sq_sum += area * area * double(distinct_n_faces[i]);
    n_total += distinct_n_faces[i];
  }
  double area_mean = area_sum / double(n_total);
  double area_sq_mean = area_sq_sum / double(n_total);
  assert(!isnan(area_mean));

  return area_sq_mean / (area_mean * area_mean) - 1.0;
}

double DualAreaError(QfeLatticeS2& lattice) {
  double area_sum = 0.0;
  double area_sq_sum = 0.0;

  for (int id = 0; id < lattice.n_distinct; id++) {
    int s = lattice.distinct_first[id];
    double site_area = 0.0;
    for (int n = 0; n < lattice.sites[s].nn; n++) {
      int l = lattice.sites[s].links[n];

      for (int i = 0; i < 2; i++) {
        // find the other two edges of this face
        int f = lattice.links[l].faces[i];
        int e = 0;
        while (lattice.faces[f].edges[e] != l) e++;
        int e1 = (e + 1) % 3;
        int e2 = (e + 2) % 3;
        int l1 = lattice.faces[f].edges[e1];
        int l2 = lattice.faces[f].edges[e2];

        // find the area associated with this face
        double sq_edge = lattice.EdgeSquared(l);
        double sq_edge_1 = lattice.EdgeSquared(l1);
        double sq_edge_2 = lattice.EdgeSquared(l2);
        double half_wt =
            (sq_edge_1 + sq_edge_2 - sq_edge) / (8.0 * lattice.FlatArea(f));

        // add to the weights of the two sites connected by this link
        site_area += 0.25 * half_wt * sq_edge;
      }
    }
    area_sum += site_area * double(lattice.distinct_n_sites[id]);
    area_sq_sum += site_area * site_area * double(lattice.distinct_n_sites[id]);
  }
  double area_mean = area_sum / double(lattice.n_sites);
  double area_sq_mean = area_sq_sum / double(lattice.n_sites);
  assert(!isnan(area_mean));

  return area_sq_mean / (area_mean * area_mean) - 1.0;
}

double DeficitAngleError(QfeLatticeS2& lattice) {
  double deficit_sum = 0.0;
  double deficit_sq_sum = 0.0;
  for (int id = 0; id < lattice.n_distinct; id++) {
    int s = lattice.distinct_first[id];

    // loop over all pairs of neighbors
    double loop_angle = 0.0;
    for (int n1 = 0; n1 < lattice.sites[s].nn; n1++) {
      int s1 = lattice.sites[s].neighbors[n1];
      for (int n2 = n1 + 1; n2 < lattice.sites[s].nn; n2++) {
        int s2 = lattice.sites[s].neighbors[n2];
        int f = lattice.FindFace(s, s1, s2);
        if (f == -1) continue;

        Vec3 v1 = (lattice.r[s1] - lattice.r[s]).normalized();
        Vec3 v2 = (lattice.r[s2] - lattice.r[s]).normalized();
        loop_angle += acos(v1.dot(v2));
      }
    }
    double deficit_angle = 2.0 * M_PI - loop_angle;
    deficit_sum += deficit_angle * lattice.distinct_n_sites[id];
    deficit_sq_sum +=
        deficit_angle * deficit_angle * lattice.distinct_n_sites[id];
  }

  double deficit_mean = deficit_sum / double(lattice.n_sites);
  double deficit_sq_mean = deficit_sq_sum / double(lattice.n_sites);
  assert(!isnan(deficit_mean));

  return deficit_sq_mean / (deficit_mean * deficit_mean) - 1.0;
}

double CombinedError(QfeLatticeS2& lattice) {
  // return FaceAreaError(lattice);
  // return DualAreaError(lattice);
  // return CircumradiusError(lattice);
  // return FaceAreaError(lattice) + CircumradiusError(lattice) +
  //        DualAreaError(lattice);
  // return DeficitAngleError(lattice);
  return FaceAreaError(lattice) + DeficitAngleError(lattice);
}

void PrintError(QfeLatticeS2& lattice) {
  double face_err = FaceAreaError(lattice);
  double cr_err = CircumradiusError(lattice);
  double dual_err = DualAreaError(lattice);
  double deficit_err = DeficitAngleError(lattice);
  double sum_err = face_err + cr_err + dual_err;
  printf("face_err: %.12e\n", face_err);
  printf("cr_err:   %.12e\n", cr_err);
  printf("dual_err: %.12e\n", dual_err);
  printf("deficit_err: %.12e\n", deficit_err);
  printf("sum_err:  %.12e\n", sum_err);
}

/// @brief This program attempts find a simplicial lattice discretization of a
/// 2-sphere such that all triangles have an equal effective lattice spacing. In
/// practice, we minimize the variance in both the triangle areas and the
/// angular deficit at each vertex. We first identify the degrees of freedom
/// which do not break polyhedral symmetry. We then use Newton's method to find
/// the minimum in the variance.
int main(int argc, const char* argv[]) {
  int q = 5;
  int k = 1;
  std::string orbit_path = "";

  if (argc > 1) q = atoi(argv[1]);
  if (argc > 2) k = atoi(argv[2]);
  if (argc > 3) orbit_path = argv[3];

  QfeLatticeS2 lattice(q, k);

  std::vector<int> dof_orbit;
  std::vector<int> dof_index;
  orbit_type.resize(lattice.n_distinct);  // type of each orbit
  for (int o = 0; o < lattice.n_distinct; o++) {
    Vec3 xi = lattice.orbit_xi[o];

    // determine the orbit type
    OrbitType type;
    if (AlmostEq(xi[0], 1.0)) {
      type = OrbitType::V;
    } else if (AlmostEq(xi[0], 0.5) && AlmostEq(xi[1], 0.5)) {
      type = OrbitType::E;
    } else if (AlmostEq(xi[0], xi[1]) && AlmostEq(xi[1], xi[2])) {
      type = OrbitType::F;
    } else if (AlmostEq(xi[2], 0.0)) {
      type = OrbitType::VE;
    } else if (AlmostEq(xi[2], xi[1])) {
      type = OrbitType::VF;
    } else if (AlmostEq(xi[1], xi[0])) {
      type = OrbitType::EF;
    } else {
      type = OrbitType::VEF;
    }

    orbit_type[o] = type;
    // printf("%04d %d\n", o, type);

    // count the number of degrees of freedom in this orbit
    if (type >= OrbitType::VE) {
      dof_orbit.push_back(o);
      dof_index.push_back(0);
    }

    if (type == OrbitType::VEF) {
      dof_orbit.push_back(o);
      dof_index.push_back(1);
    }
  }
  int n_dof = dof_orbit.size();
  printf("n_dof: %d\n", n_dof);

  // find distinct faces
  for (int f = 0; f < lattice.n_faces; f++) {
    int id = lattice.face_orbit[f];
    while (id >= distinct_n_faces.size()) {
      distinct_n_faces.push_back(0);
      distinct_first_face.push_back(f);
    }

    distinct_n_faces[id]++;
  }

  // printf("face degeneracies:\n");
  // for (int i = 0; i < distinct_n_faces.size(); i++) {
  //   printf("%d %d\n", i, distinct_n_faces[i]);
  // }

  // generate the set of primary sites
  for (int id = 0; id < lattice.n_distinct; id++) {
    int s = lattice.distinct_first[id];
    primary_sites.insert(s);
    for (int n = 0; n < lattice.sites[s].nn; n++) {
      int s1 = lattice.sites[s].neighbors[n];
      primary_sites.insert(s1);
    }
  }

  for (int i = 0; i < distinct_first_face.size(); i++) {
    int f = distinct_first_face[i];
    primary_sites.insert(lattice.faces[f].sites[0]);
    primary_sites.insert(lattice.faces[f].sites[1]);
    primary_sites.insert(lattice.faces[f].sites[2]);
  }
  printf("# of primary sites: %lu\n", primary_sites.size());

  // compute the initial error
  PrintError(lattice);
  double error_sum = CombinedError(lattice);
  double old_error = error_sum;

  double delta = 1.0e-5;
  double delta_sq = delta * delta;
  int mu = 0;  // preconditioning value
  for (int n = 0; n < 10000; n++) {
    Mat A = Mat::Zero(n_dof, n_dof);
    Mat b = Vec::Zero(n_dof);

    double F0 = CombinedError(lattice);

    for (int d1 = 0; d1 < n_dof; d1++) {
      int o1 = dof_orbit[d1];
      int dof1 = dof_index[d1];
      int s1 = lattice.distinct_first[o1];

      double base_value1 = lattice.orbit_xi[o1](dof1);
      double plus_value1 = base_value1 + delta;
      double minus_value1 = base_value1 - delta;

      lattice.orbit_xi[o1](dof1) = plus_value1;
      UpdateOrbit(lattice, o1);
      double Fp = CombinedError(lattice);

      lattice.orbit_xi[o1](dof1) = minus_value1;
      UpdateOrbit(lattice, o1);
      double Fm = CombinedError(lattice);

      // calculate first derivative
      b(d1) = 0.5 * (Fp - Fm) / delta;

      // calculate diagonal term for 2nd derivative
      A(d1, d1) = (Fp + Fm - 2.0 * F0) / delta_sq;

      // calculate the off-diagonal 2nd derivative terms
      for (int d2 = d1 + 1; d2 < n_dof; d2++) {
        int o2 = dof_orbit[d2];
        int dof2 = dof_index[d2];

        // skip unless id1 and id2 are neighbors
        bool is_neighbor = false;
        for (int nn = 0; nn < lattice.sites[s1].nn; nn++) {
          int s_n = lattice.sites[s1].neighbors[nn];
          if (lattice.sites[s_n].id == o2) {
            is_neighbor = true;
            break;
          }
        }
        if (!is_neighbor) continue;

        double base_value2 = lattice.orbit_xi[o2](dof2);
        double plus_value2 = base_value2 + delta;
        double minus_value2 = base_value2 - delta;

        lattice.orbit_xi[o1](dof1) = plus_value1;
        UpdateOrbit(lattice, o1);
        lattice.orbit_xi[o2](dof2) = plus_value2;
        UpdateOrbit(lattice, o2);
        double Fpp = CombinedError(lattice);
        lattice.orbit_xi[o2](dof2) = minus_value2;
        UpdateOrbit(lattice, o2);
        double Fpm = CombinedError(lattice);

        lattice.orbit_xi[o1](dof1) = minus_value1;
        UpdateOrbit(lattice, o1);
        lattice.orbit_xi[o2](dof2) = plus_value2;
        UpdateOrbit(lattice, o2);
        double Fmp = CombinedError(lattice);
        lattice.orbit_xi[o2](dof2) = minus_value2;
        UpdateOrbit(lattice, o2);
        double Fmm = CombinedError(lattice);

        lattice.orbit_xi[o2](dof2) = base_value2;
        UpdateOrbit(lattice, o2);

        A(d1, d2) = 0.25 * (Fpp - Fpm - Fmp + Fmm) / delta_sq;

        // matrix is symmetric
        A(d2, d1) = A(d1, d2);
      }

      lattice.orbit_xi[o1](dof1) = base_value1;
      UpdateOrbit(lattice, o1);
    }

    if (mu > 0) mu--;     // try to decrease mu by 1 once per iteration
    double lambda = 1.0;  // keep this fixed at 1.0

    std::vector<double> old_dof(n_dof);
    for (int d = 0; d < n_dof; d++) {
      int o = dof_orbit[d];
      int dof = dof_index[d];
      old_dof[d] = lattice.orbit_xi[o](dof);
    }

    while (true) {
      // compute the solution
      Eigen::ConjugateGradient<Mat> cg;
      Mat A_precond = A + Mat::Identity(n_dof, n_dof) * double(mu);

      cg.compute(A_precond);
      assert(cg.info() == Eigen::Success);
      Vec x = cg.solve(b);

      // update the barycentric coordinates of each orbit
      for (int d = 0; d < n_dof; d++) {
        int o = dof_orbit[d];
        int dof = dof_index[d];
        lattice.orbit_xi[o](dof) = old_dof[d] - x(d) * lambda;
      }

      // apply the new positions to all of the orbits
      for (int d = 0; d < n_dof; d++) {
        if (dof_index[d] != 0) continue;
        int o = dof_orbit[d];
        UpdateOrbit(lattice, o);
      }

      // compute the new error
      error_sum = CombinedError(lattice);

      if (error_sum < old_error) break;
      // printf("%04d %02d %.2e %.12e\n", n, mu, lambda, error_sum);
      assert(!isnan(error_sum));

      mu++;
      if (mu > 500) break;
      A += Mat::Identity(n_dof, n_dof);
    }

    // check the error
    double delta_err = (old_error - error_sum) / old_error;
    old_error = error_sum;
    printf("%04d %02d %.4f %.12e %.12e\n", n, mu, lambda, error_sum, delta_err);
    if (mu > 500) break;
    if (delta_err < 1.0e-10 || error_sum < 1.0e-14) break;
  }

  PrintError(lattice);
  // lattice.UpdateOrbits();
  // lattice.PrintCoordinates();

  // save orbit data to file
  if (!orbit_path.empty()) {
    FILE* orbit_file = fopen(orbit_path.c_str(), "w");
    assert(orbit_file != nullptr);
    lattice.WriteOrbits(orbit_file);
    fclose(orbit_file);
  }

  return 0;
}
