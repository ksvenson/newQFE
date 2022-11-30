// s2_std_uniform.cc

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/IterativeLinearSolvers>
#include <cmath>
#include <cstdio>
#include <iostream>

#include "s2.h"

typedef double Real;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> Mat;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> Vec;
typedef Eigen::Quaternion<Real> Quat;
typedef Eigen::Vector3<Real> Vec3;

struct GrpElem {
  int id;
  Quat q;
  bool inv;
};

enum OrbitType { V, E, F, VE, VF, EF, VEF };

struct Orbit {
  int id;
  OrbitType type;
  int n_dof;
  Real dof[2];
};

std::vector<Orbit> orbits;
std::vector<int> site_g;
std::vector<GrpElem> G;
std::vector<int> distinct_face;
std::vector<int> distinct_n_faces;
std::vector<int> distinct_first_face;

Vec3 poly_v[3];  // primary triangle vertices

Vec3 GrpMult(Vec3& v, GrpElem& g) {
  Vec3 gr = g.q * v;
  if (g.inv) gr *= -1.0;
  return gr;
}

Vec3 CalcFirstVertex(OrbitType type, Real dof[]) {
  Real xi[3];
  xi[0] = 0.0;
  xi[1] = 0.0;
  xi[2] = 0.0;

  switch (type) {
    case OrbitType::V:
      xi[2] = 1.0;
      break;
    case OrbitType::E:
      xi[2] = 0.5;
      xi[1] = 0.5;
      break;
    case OrbitType::F:
      xi[2] = 1.0 / 3.0;
      xi[1] = xi[2];
      xi[0] = xi[2];
      break;
    case OrbitType::VE:
      xi[2] = dof[0];
      xi[1] = 1.0 - dof[0];
      break;
    case OrbitType::VF:
      xi[2] = dof[0];
      xi[1] = 0.5 - 0.5 * xi[2];
      break;
    case OrbitType::EF:
      xi[2] = dof[0];
      xi[1] = xi[2];
      break;
    case OrbitType::VEF:
      xi[2] = dof[0];
      xi[1] = dof[1];
      break;
    default:
      printf("invalid orbit\n");
      exit(1);
  }

  xi[0] = 1.0 - (xi[2] + xi[1]);

  Vec3 v = Vec3::Zero();
  for (int i = 0; i < 3; i++) {
    v += xi[i] * poly_v[i];
  }
  return v.normalized();
}

void UpdateOrbit(QfeLatticeS2& lattice, int o) {
  Orbit* this_orbit = &orbits[o];

  Vec3 v_first = CalcFirstVertex(this_orbit->type, this_orbit->dof);
  for (int s = 0; s < lattice.n_sites; s++) {
    if (lattice.sites[s].id != this_orbit->id) continue;
    lattice.r[s] = GrpMult(v_first, G[site_g[s]]);
  }
}

bool VectorsAreEqual(Vec3& r1, Vec3& r2) {
  return (fabs(r1.dot(r2) - 1.0) < 1.0e-12);
}

double FaceAreaError(QfeLatticeS2& lattice) {
  double area_sum = 0.0;
  std::vector<double> area_list(distinct_first_face.size());
  int n_total = 0;
  for (int i = 0; i < distinct_first_face.size(); i++) {
    double area = lattice.FlatArea(distinct_first_face[i]);
    area_list[i] = area;
    area_sum += area * double(distinct_n_faces[i]);
    n_total += distinct_n_faces[i];
  }
  double area_mean = area_sum / double(n_total);
  assert(!isnan(area_mean));

  double error_sum = 0.0;
  for (int i = 0; i < distinct_first_face.size(); i++) {
    double area_err = 1.0 - area_list[i] / area_mean;
    error_sum += pow(area_err, 2.0) * double(distinct_n_faces[i]);
  }

  return error_sum / double(n_total);
}

double CircumradiusError(QfeLatticeS2& lattice) {
  double area_sum = 0.0;
  std::vector<double> area_list(distinct_first_face.size());
  int n_total = 0;
  for (int i = 0; i < distinct_first_face.size(); i++) {
    int f = distinct_first_face[i];
    int s = lattice.faces[f].sites[0];
    Vec3 cc = lattice.FaceCircumcenter(distinct_first_face[i]);
    double cr = (cc - lattice.r[s]).norm();
    double area = cr * cr;
    area_list[i] = area;
    area_sum += area * double(distinct_n_faces[i]);
    n_total += distinct_n_faces[i];
  }
  double area_mean = area_sum / double(n_total);

  double error_sum = 0.0;
  for (int i = 0; i < distinct_first_face.size(); i++) {
    double area_err = 1.0 - area_list[i] / area_mean;
    error_sum += pow(area_err, 2.0) * double(distinct_n_faces[i]);
  }

  return error_sum / double(n_total);
}

double DualAreaError(QfeLatticeS2& lattice) {
  double area_sum = 0.0;
  std::vector<double> area_list(lattice.n_distinct);

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
    area_list[id] = site_area;
    area_sum += site_area * double(lattice.distinct_n_sites[id]);
  }
  double area_mean = area_sum / double(lattice.n_sites);

  double error_sum = 0.0;
  for (int id = 0; id < lattice.n_distinct; id++) {
    double area_err = 1.0 - area_list[id] / area_mean;
    error_sum += pow(area_err, 2.0) * lattice.distinct_n_sites[id];
  }

  return error_sum / double(lattice.n_sites);
}

double CombinedError(QfeLatticeS2& lattice) {
  // return FaceAreaError(lattice);
  // return DualAreaError(lattice);
  // return CircumradiusError(lattice);
  return FaceAreaError(lattice) + CircumradiusError(lattice) +
         DualAreaError(lattice);
}

void PrintError(QfeLatticeS2& lattice) {
  double face_err = FaceAreaError(lattice);
  double cr_err = CircumradiusError(lattice);
  double dual_err = DualAreaError(lattice);
  double sum_err = face_err + cr_err + dual_err;
  printf("face_err: %.12e\n", face_err);
  printf("cr_err:   %.12e\n", cr_err);
  printf("dual_err: %.12e\n", dual_err);
  printf("sum_err:  %.12e\n", sum_err);
}

/**
 * @brief This program attempts find a simplicial lattice discretization of
 * a 2-sphere such that all triangles have equal circumradius. In practice,
 * we minimize the function
 *
 * f(x_i) = sum_tri [(1 - (R_tri / R_mean)^2)^2 (1 - A_tri / A_mean)^2]
 *
 * where x_i are the positions of the lattice sites, R_tri is a triangle's
 * circumradius, and R_mean is the mean circumradius
 *
 * R_mean = 1 / N_tri sum_tri R_tri
 *
 * We parameterize the coordinates of each point in spherical coordinates,
 * theta and phi. We then use Newton's method to solve the non-linear
 * equation A x = b, where x is a vector of independent degrees of freedom
 * of the site coordinates and b is the gradient of f(x_i).
 **/

int main(int argc, const char* argv[]) {
  int q = 5;

  char base_path[200];
  sprintf(base_path, "%s", "s2_std/q5k3");
  if (argc > 1) {
    sprintf(base_path, "%s", argv[1]);
  }

  QfeLatticeS2 lattice(0);

  // read site coordinates
  char grid_path[200];
  sprintf(grid_path, "%s_std.dat", base_path);
  FILE* grid_file = fopen(grid_path, "r");
  assert(grid_file != nullptr);
  lattice.ReadLattice(grid_file);
  fclose(grid_file);

  // read orbit data
  char orbit_path[200];
  sprintf(orbit_path, "%s_orbit.dat", base_path);
  FILE* orbit_file = fopen(orbit_path, "r");
  assert(orbit_file != nullptr);
  orbits.resize(lattice.n_distinct);

  std::vector<int> dof_orbit;
  std::vector<int> dof_index;
  for (int o = 0; o < lattice.n_distinct; o++) {
    int index;
    OrbitType type;
    Real dof[2];
    fscanf(orbit_file, "%d %d %lf %lf", &index, &type, &dof[0], &dof[1]);

    // create the orbit
    orbits[o].id = index;
    orbits[o].type = type;
    orbits[o].n_dof = 0;
    orbits[o].dof[0] = dof[0];
    orbits[o].dof[1] = dof[1];
    assert(index == o);

    // set the number of degrees of freedom base on the orbit type
    if (type >= OrbitType::VE) orbits[o].n_dof++;
    if (type >= OrbitType::VEF) orbits[o].n_dof++;

    for (int i = 0; i < orbits[o].n_dof; i++) {
      dof_orbit.push_back(o);
      dof_index.push_back(i);
    }
  }
  fclose(orbit_file);
  int n_dof = dof_orbit.size();
  printf("n_dof: %d\n", n_dof);

  // vertices of first polytope
  QfeLatticeS2 base_lattice(q);
  poly_v[0] = base_lattice.r[0];
  poly_v[1] = base_lattice.r[1];
  poly_v[2] = base_lattice.r[2];

  // load the symmetry group data
  char grp_path[200];
  sprintf(grp_path, "grp_s2_q%d.dat", q);
  FILE* grp_file = fopen(grp_path, "r");
  assert(grp_file != nullptr);

  // read the symmetry group data
  while (!feof(grp_file)) {
    int index;
    int inv;
    Real q_w, q_x, q_y, q_z;
    fscanf(grp_file, "%d %d %lf %lf %lf %lf", &index, &inv, &q_w, &q_x, &q_y,
           &q_z);
    GrpElem g;
    g.id = index;
    g.q = Quat(q_w, q_x, q_y, q_z);
    g.inv = (inv == -1);
    G.push_back(g);
  }
  fclose(grp_file);

  // find the group element for each site
  site_g.resize(lattice.n_sites);
  for (int s = 0; s < lattice.n_sites; s++) {
    int id = lattice.sites[s].id;
    Orbit* this_orbit = &orbits[id];
    Vec3 r_orbit = CalcFirstVertex(this_orbit->type, this_orbit->dof);

    // find the appropriate group element
    for (int g = 0; g < G.size(); g++) {
      Vec3 gr = GrpMult(r_orbit, G[g]);
      if (VectorsAreEqual(lattice.r[s], gr)) {
        site_g[s] = g;
        break;
      }
    }
    // printf("%04d %04d\n", s, site_g[s]);
  }

  // find distinct faces
  std::map<std::string, int> face_map;
  double face_area = 0.0;
  double cr_sum = 0.0;
  double cr2_sum = 0.0;
  for (int f = 0; f < lattice.n_faces; f++) {
    face_area += lattice.FlatArea(f);
    Vec3 face_cc = lattice.FaceCircumcenter(f);
    double cr = (face_cc - lattice.r[lattice.faces[f].sites[0]]).norm();
    cr_sum += cr;
    cr2_sum += cr * cr;

    std::vector<int> site_id(3);
    site_id[0] = lattice.sites[lattice.faces[f].sites[0]].id;
    site_id[1] = lattice.sites[lattice.faces[f].sites[1]].id;
    site_id[2] = lattice.sites[lattice.faces[f].sites[2]].id;
    std::sort(site_id.begin(), site_id.end());
    char face_key[200];
    sprintf(face_key, "%d_%d_%d_%.6f", site_id[0], site_id[1], site_id[2],
            lattice.FlatArea(f));
    if (face_map.find(face_key) == face_map.end()) {
      face_map[face_key] = distinct_first_face.size();
      distinct_first_face.push_back(f);
      distinct_n_faces.push_back(0);
    }
    int face_id = face_map[face_key];
    distinct_face.push_back(face_id);
    distinct_n_faces[face_id]++;
  }
  double cr_mean = cr_sum / double(lattice.n_faces);
  double cr2_mean = cr2_sum / double(lattice.n_faces);
  double cr_stdev =
      sqrt((cr2_mean - cr_mean * cr_mean) / double(lattice.n_faces));
  printf("face_area: %.12f\n", face_area);
  printf("cr_mean: %.12f\n", cr_mean);
  printf("cr_stdev:  %.12e\n", cr_stdev);

  // printf("face degeneracies:\n");
  // for (int i = 0; i < distinct_n_faces.size(); i++) {
  //   printf("%d %d\n", i, distinct_n_faces[i]);
  // }

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

      double base_value1 = orbits[o1].dof[dof1];
      double plus_value1 = base_value1 + delta;
      double minus_value1 = base_value1 - delta;

      orbits[o1].dof[dof1] = plus_value1;
      UpdateOrbit(lattice, o1);
      double Fp = CombinedError(lattice);

      orbits[o1].dof[dof1] = minus_value1;
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

        double base_value2 = orbits[o2].dof[dof2];
        double plus_value2 = base_value2 + delta;
        double minus_value2 = base_value2 - delta;

        orbits[o1].dof[dof1] = plus_value1;
        UpdateOrbit(lattice, o1);
        orbits[o2].dof[dof2] = plus_value2;
        UpdateOrbit(lattice, o2);
        double Fpp = CombinedError(lattice);
        orbits[o2].dof[dof2] = minus_value2;
        UpdateOrbit(lattice, o2);
        double Fpm = CombinedError(lattice);

        orbits[o1].dof[dof1] = minus_value1;
        UpdateOrbit(lattice, o1);
        orbits[o2].dof[dof2] = plus_value2;
        UpdateOrbit(lattice, o2);
        double Fmp = CombinedError(lattice);
        orbits[o2].dof[dof2] = minus_value2;
        UpdateOrbit(lattice, o2);
        double Fmm = CombinedError(lattice);

        orbits[o2].dof[dof2] = base_value2;
        UpdateOrbit(lattice, o2);

        A(d1, d2) = 0.25 * (Fpp - Fpm - Fmp + Fmm) / delta_sq;

        // matrix is symmetric
        A(d2, d1) = A(d1, d2);
      }

      orbits[o1].dof[dof1] = base_value1;
      UpdateOrbit(lattice, o1);
    }

    if (mu > 0) mu--;     // try to decrease mu by 1 once per iteration
    double lambda = 1.0;  // keep this fixed at 1.0

    std::vector<double> old_dof(n_dof);
    for (int d = 0; d < n_dof; d++) {
      int o = dof_orbit[d];
      int dof = dof_index[d];
      old_dof[d] = orbits[o].dof[dof];
    }

    while (true) {
      // compute the solution
      Eigen::ConjugateGradient<Mat> cg;
      Mat A_precond = A + Mat::Identity(n_dof, n_dof) * double(mu);

      cg.compute(A_precond);
      assert(cg.info() == Eigen::Success);
      Vec x = cg.solve(b);

      for (int d = 0; d < n_dof; d++) {
        int o = dof_orbit[d];
        int dof = dof_index[d];
        orbits[o].dof[dof] = old_dof[d] - x(d) * lambda;
      }

      for (int o = 0; o < orbits.size(); o++) {
        UpdateOrbit(lattice, o);
      }

      // compute the new error
      error_sum = CombinedError(lattice);

      if (error_sum < old_error) break;
      if (isnan(error_sum)) exit(1);

      // if error increased, increase preconditioner value
      mu++;
      if (mu == 500) break;
      A += Mat::Identity(n_dof, n_dof);
    }

    // check the error
    double delta_err = (old_error - error_sum) / old_error;
    old_error = error_sum;
    printf("%04d %03d %.12e %.12e\n", n, mu, error_sum, delta_err);
    if (mu == 500) break;
    if (delta_err < 1.0e-10 || error_sum < 1.0e-14) break;
  }

  PrintError(lattice);
  lattice.UpdateWeights();
  // lattice.PrintCoordinates();

  // save lattice to file
  char lattice_path[200];
  sprintf(lattice_path, "%s_uniform.dat", base_path);
  FILE* lattice_file = fopen(lattice_path, "w");
  assert(lattice_file != nullptr);
  lattice.WriteLattice(lattice_file);
  fclose(lattice_file);

  return 0;
}
