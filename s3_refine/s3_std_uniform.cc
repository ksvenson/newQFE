// s3_std_uniform.cc

#include <cmath>
#include <cstdio>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/IterativeLinearSolvers>
#include "s3.h"

typedef double Real;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> Mat;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> Vec;
typedef Eigen::Quaternion<Real> Quat;
typedef Eigen::Vector4<Real> Vec4;

// vectors are stored (x,y,z,w) but quaternions are (w,x,y,z)
// it's annoying but this is how Eigen works
#define QuatToVec(q) Vec4(q.x(), q.y(), q.z(), q.w())
#define VecToQuat(v) Quat(v.w(), v.x(), v.y(), v.z())

struct GrpElem {
  int id;
  Quat l;
  Quat r;
  bool star;
};

enum OrbitType {
  V, E, F, C,
  VE, VF, VC, EF, EC, FC,
  VEF, VEC, VFC, EFC,
  VEFC
};

struct Orbit {
  int id;
  OrbitType type;
  int n_dof;
  Real dof[3];
};

std::vector<Orbit> orbits;
std::vector<int> site_g;
std::vector<GrpElem> G;
std::vector<int> distinct_cell;
std::vector<int> distinct_n_cells;
std::vector<int> distinct_first_cell;

Vec4 poly_v[4];  // primary tetrahedron vertices

Vec4 GrpMult(Vec4& v, GrpElem& g) {
  Quat q = VecToQuat(v);
  q = g.l * (g.star ? q.conjugate() : q) * g.r;
  return QuatToVec(q).normalized();
}

Vec4 CalcFirstVertex(OrbitType type, Real dof[]) {
  Real xi[4];
  xi[0] = 0.0;
  xi[1] = 0.0;
  xi[2] = 0.0;
  xi[3] = 0.0;

  switch (type) {
    case OrbitType::V: xi[3] = 1.0; break;
    case OrbitType::E: xi[3] = 0.5; xi[2] = 0.5; break;
    case OrbitType::F: xi[3] = 1.0 / 3.0; xi[2] = xi[3]; xi[1] = xi[3]; break;
    case OrbitType::C: xi[3] = 0.25; xi[2] = xi[3]; xi[1] = xi[3]; break;
    case OrbitType::VE: xi[3] = dof[0]; xi[2] = 1.0 - dof[0]; break;
    case OrbitType::VF: xi[3] = dof[0]; xi[2] = 0.5 - 0.5 * xi[3]; xi[1] = xi[2]; break;
    case OrbitType::VC: xi[3] = dof[0]; xi[2] = (1.0 - xi[3]) / 3.0; xi[1] = xi[2]; break;
    case OrbitType::EF: xi[3] = dof[0]; xi[2] = xi[3]; xi[1] = 1.0 - 2.0 * xi[3]; break;
    case OrbitType::EC: xi[3] = dof[0]; xi[2] = xi[3]; xi[1] = 0.5 - xi[3]; break;
    case OrbitType::FC: xi[3] = dof[0]; xi[2] = xi[3]; xi[1] = xi[3]; break;
    case OrbitType::VEF: xi[3] = dof[0]; xi[2] = dof[1]; xi[1] = 1.0 - (xi[3] + xi[2]); break;
    case OrbitType::VEC: xi[3] = dof[0]; xi[2] = dof[1]; xi[1] = 0.5 - 0.5 * (xi[3] + xi[2]); break;
    case OrbitType::VFC: xi[3] = dof[0]; xi[2] = dof[1]; xi[1] = xi[2]; break;
    case OrbitType::EFC: xi[3] = dof[0]; xi[2] = xi[3]; xi[1] = dof[1]; break;
    case OrbitType::VEFC: xi[3] = dof[0]; xi[2] = dof[1]; xi[1] = dof[2]; break;
    default: printf("invalid orbit\n"); exit(1);
  }

  xi[0] = 1.0 - xi[1] - xi[2] - xi[3];

  Vec4 v = Vec4::Zero();
  for (int i = 0; i < 4; i++) {
    v += xi[i] * poly_v[i];
  }
  return v.normalized();
}

void UpdateOrbit(QfeLatticeS3& lattice, int o) {
  Orbit* this_orbit = &orbits[o];

  Vec4 v_first = CalcFirstVertex(this_orbit->type, this_orbit->dof);
  for (int s = 0; s < lattice.n_sites; s++) {
    if (lattice.sites[s].id != this_orbit->id) continue;
    lattice.r[s] = GrpMult(v_first, G[site_g[s]]);
  }
}

bool VectorsAreEqual(Vec4& r1, Vec4& r2) {
  return (fabs(r1.dot(r2) - 1.0) < 1.0e-12);
}

double CellVolumeError(QfeLatticeS3& lattice) {
  double vol_sum = 0.0;
  std::vector<double> vol_list(distinct_first_cell.size());
  int n_total = 0;
  for (int i = 0; i < distinct_first_cell.size(); i++) {
    double vol = lattice.CellVolume(distinct_first_cell[i]);
    vol_list[i] = vol;
    vol_sum += vol * double(distinct_n_cells[i]);
    n_total += distinct_n_cells[i];
  }
  double vol_mean = vol_sum / double(n_total);
  assert(!isnan(vol_mean));

  double error_sum = 0.0;
  for (int i = 0; i < distinct_first_cell.size(); i++) {
    double vol_err = 1.0 - vol_list[i] / vol_mean;
    error_sum += pow(vol_err, 2.0) * double(distinct_n_cells[i]);
  }

  return error_sum / double(n_total);
}

double CircumradiusError(QfeLatticeS3& lattice) {
  double vol_sum = 0.0;
  std::vector<double> vol_list(distinct_first_cell.size());
  int n_total = 0;
  for (int i = 0; i < distinct_first_cell.size(); i++) {
    int c = distinct_first_cell[i];
    int s = lattice.cells[c].sites[0];
    Vec4 cc = lattice.CellCircumcenter(distinct_first_cell[i]);
    double cr = (cc - lattice.r[s]).norm();
    double vol = cr * cr * cr;
    vol_list[i] = vol;
    vol_sum += vol * double(distinct_n_cells[i]);
    n_total += distinct_n_cells[i];
  }
  double vol_mean = vol_sum / double(n_total);

  double error_sum = 0.0;
  for (int i = 0; i < distinct_first_cell.size(); i++) {
    double vol_err = 1.0 - vol_list[i] / vol_mean;
    error_sum += pow(vol_err, 2.0) * double(distinct_n_cells[i]);
  }

  return error_sum / double(n_total);
}

double DualVolumeError(QfeLatticeS3& lattice) {

  // adjust octahedral centers
  for (int s = 0; s < lattice.n_sites; s++) {
    if (lattice.sites[s].nn != 6) continue;
    Vec4 r_mean = Vec4::Zero();
    for (int n = 0; n < 6; n++) {
      r_mean += lattice.r[lattice.sites[s].neighbors[n]];
    }
    lattice.r[s] = r_mean.normalized();
  }

  double vol_sum = 0.0;
  std::vector<double> vol_list(lattice.n_distinct);

  for (int o = 0; o < lattice.n_distinct; o++) {
    double site_vol = 0.0;
    int s1 = lattice.distinct_first[o];
    for (int n = 0; n < lattice.sites[s1].nn; n++) {
      int l = lattice.sites[s1].links[n];
      int s2 = lattice.sites[s1].neighbors[n];
      for (int lc = 0; lc < lattice.links[l].n_faces; lc++) {
        int f = lattice.links[l].faces[lc];
        for (int fc = 0; fc < 2; fc++) {
          int c = lattice.faces[f].cells[fc];

          Vec4 cell_r[5];

          cell_r[0] = Vec4::Zero();
          cell_r[1] = lattice.r[lattice.cells[c].sites[0]];
          cell_r[2] = lattice.r[lattice.cells[c].sites[1]];
          cell_r[3] = lattice.r[lattice.cells[c].sites[2]];
          cell_r[4] = lattice.r[lattice.cells[c].sites[3]];

          Eigen::Matrix<double, 5, 5> CM;
          for (int i = 0; i < 5; i++) {
            CM(i,i) = 0.0;
            for (int j = i + 1; j < 5; j++) {
              CM(i,j) = (cell_r[i] - cell_r[j]).squaredNorm();
              CM(j,i) = CM(i,j);
            }
          }

          Eigen::Vector<double, 5> cell_lhs(1.0, 0.0, 0.0, 0.0, 0.0);
          Eigen::Vector<double, 5> cell_xi = CM.inverse() * cell_lhs;
          double cell_cr_sq = -cell_xi(0) / 2.0;

          // find the sites for this link
          int i = 1;
          while (lattice.cells[c].sites[i - 1] != s1) i++;
          int j = 1;
          while (lattice.cells[c].sites[j - 1] != s2) j++;

          // find the other two corners of the cell
          int k = 1;
          while (k == i || k == j) k++;
          int l = k + 1;
          while (l == i || l == j) l++;

          double x_ijk = 2.0 * (CM(i,j) * CM(i,k) + CM(i,j) * CM(j,k) + CM(i,k) * CM(j,k)) - (CM(i,j) * CM(i,j) + CM(i,k) * CM(i,k) + CM(j,k) * CM(j,k));
          double x_ijl = 2.0 * (CM(i,j) * CM(i,l) + CM(i,j) * CM(j,l) + CM(i,l) * CM(j,l)) - (CM(i,j) * CM(i,j) + CM(i,l) * CM(i,l) + CM(j,l) * CM(j,l));

          double A_tri_ijk = 0.25 * sqrt(x_ijk);
          double A_tri_ijl = 0.25 * sqrt(x_ijl);
          double dual_ijk = CM(i,k) + CM(j,k) - CM(i,j);
          double dual_ijl = CM(i,l) + CM(j,l) - CM(i,j);

          double h_ijk = sqrt(cell_cr_sq - CM(i,j) * CM(i,k) * CM(j,k) / x_ijk);
          double h_ijl = sqrt(cell_cr_sq - CM(i,j) * CM(i,l) * CM(j,l) / x_ijl);

          if (cell_xi(l) < 0.0) h_ijk *= -1.0;
          if (cell_xi(k) < 0.0) h_ijl *= -1.0;

          double wt = (dual_ijk * h_ijk / A_tri_ijk + dual_ijl * h_ijl / A_tri_ijl) / 16.0;
          double vol = wt * CM(i,j) / 6.0;

          site_vol += vol;
        }
      }
    }
    vol_list[o] = site_vol;
    vol_sum += site_vol * double(lattice.distinct_n_sites[o]);
  }
  double vol_mean = vol_sum / double(lattice.n_sites);

  double error_sum = 0.0;
  for (int o = 0; o < lattice.n_distinct; o++) {
    double vol_err = 1.0 - vol_list[o] / vol_mean;
    error_sum += pow(vol_err, 2.0) * double(lattice.distinct_n_sites[o]);
  }

  return error_sum / double(lattice.n_sites);
}

double CombinedError(QfeLatticeS3& lattice) {
  return DualVolumeError(lattice);
  // return CellVolumeError(lattice) + DualVolumeError(lattice);
  // return CellVolumeError(lattice) + CircumradiusError(lattice); // + DualVolumeError(lattice);
}

void PrintError(QfeLatticeS3& lattice) {
  double cell_err = CellVolumeError(lattice);
  double cr_err = CircumradiusError(lattice);
  double dual_err = DualVolumeError(lattice);
  double sum_err = cell_err + cr_err + dual_err;
  printf("cell_err: %.12e\n", cell_err);
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
  sprintf(base_path, "%s", "s3_std/q5k3");
  if (argc > 1) {
    sprintf(base_path, "%s", argv[1]);
  }

  QfeLatticeS3 lattice(0);

  // read site coordinates
  char grid_path[200];
  sprintf(grid_path, "%s_grid.dat", base_path);
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
    Real dof[3];
    fscanf(orbit_file, "%d %d %lf %lf %lf", \
      &index, &type, &dof[0], &dof[1], &dof[2]);

    // create the orbit
    orbits[o].id = index;
    orbits[o].type = type;
    orbits[o].n_dof = 0;
    orbits[o].dof[0] = dof[0];
    orbits[o].dof[1] = dof[1];
    orbits[o].dof[2] = dof[2];
    assert(index == o);

    // don't include orbits defined by octahedral centers
    if (lattice.sites[lattice.distinct_first[index]].nn == 6) continue;

    // set the number of degrees of freedom base on the orbit type
    if (type >= OrbitType::VE) orbits[o].n_dof++;
    if (type >= OrbitType::VEF) orbits[o].n_dof++;
    if (type == OrbitType::VEFC) orbits[o].n_dof++;

    for (int i = 0; i < orbits[o].n_dof; i++) {
      dof_orbit.push_back(o);
      dof_index.push_back(i);
    }
  }
  fclose(orbit_file);
  int n_dof = dof_orbit.size();
  printf("n_dof: %d\n", n_dof);

  // vertices of first polytope
  Vec4 poly_v1, poly_v2, poly_v3, poly_v4;

  if (q == 4) {
    poly_v[0] = Vec4(0.0, 0.0, 0.0, 1.0);
    poly_v[1] = Vec4(1.0, 0.0, 0.0, 0.0);
    poly_v[2] = Vec4(0.0, 1.0, 0.0, 0.0);
    poly_v[3] = Vec4(0.0, 0.0, 1.0, 0.0);

  } else if (q == 5) {
    const Real alpha = 0.80901699437494742410L;
    const Real beta = 0.30901699437494742410L;
    poly_v[0] = Vec4(0.0, 0.0, 0.0, 1.0);
    poly_v[1] = Vec4(0.5, 0.0, beta, alpha);
    poly_v[2] = Vec4(0.5, 0.0, -beta, alpha);
    poly_v[3] = Vec4(beta, 0.5, 0.0, alpha);
  }

  // load the symmetry group data
  char grp_path[200];
  sprintf(grp_path, "grp_s3_q%d.dat", q);
  FILE* grp_file = fopen(grp_path, "r");
  assert(grp_file != nullptr);

  // read the symmetry group data
  while (!feof(grp_file)) {
    int index;
    int inv;
    Real ql_w, ql_x, ql_y, ql_z;
    Real qr_w, qr_x, qr_y, qr_z;
    fscanf(grp_file, "%d %d %lf %lf %lf %lf %lf %lf %lf %lf", \
      &index, &inv, \
      &ql_w, &ql_x, &ql_y, &ql_z, \
      &qr_w, &qr_x, &qr_y, &qr_z);
    GrpElem g;
    g.id = index;
    g.l = Quat(ql_w, ql_x, ql_y, ql_z);
    g.r = Quat(qr_w, qr_x, qr_y, qr_z);
    g.star = (inv == -1);
    G.push_back(g);
  }
  fclose(grp_file);

  // find the group element for each site
  site_g.resize(lattice.n_sites);
  for (int s = 0; s < lattice.n_sites; s++) {
    int id = lattice.sites[s].id;
    Orbit* this_orbit = &orbits[id];
    Vec4 r_orbit = CalcFirstVertex(this_orbit->type, this_orbit->dof);

    // find the appropriate group element
    for (int g = 0; g < G.size(); g++) {
      Vec4 gr = GrpMult(r_orbit, G[g]);
      if (VectorsAreEqual(lattice.r[s], gr)) {
        site_g[s] = g;
        break;
      }
    }
    // printf("%04d %04d\n", s, site_g[s]);
  }

  // find distinct cells
  std::map<std::string, int> cell_map;
  double cell_vol = 0.0;
  double cr_sum = 0.0;
  double cr2_sum = 0.0;
  for (int c = 0; c < lattice.n_cells; c++) {
    cell_vol += lattice.CellVolume(c);
    Vec4 cell_cc = lattice.CellCircumcenter(c);
    double cr = (cell_cc - lattice.r[lattice.cells[c].sites[0]]).norm();
    cr_sum += cr;
    cr2_sum += cr * cr;

    // skip octahedral cells
    if (lattice.sites[lattice.cells[c].sites[0]].nn == 6) continue;
    if (lattice.sites[lattice.cells[c].sites[1]].nn == 6) continue;
    if (lattice.sites[lattice.cells[c].sites[2]].nn == 6) continue;
    if (lattice.sites[lattice.cells[c].sites[3]].nn == 6) continue;

    std::vector<int> site_id(4);
    site_id[0] = lattice.sites[lattice.cells[c].sites[0]].id;
    site_id[1] = lattice.sites[lattice.cells[c].sites[1]].id;
    site_id[2] = lattice.sites[lattice.cells[c].sites[2]].id;
    site_id[3] = lattice.sites[lattice.cells[c].sites[3]].id;
    std::sort(site_id.begin(), site_id.end());
    double cell_vol = lattice.CellVolume(c);
    char cell_key[200];
    sprintf(cell_key, "%d_%d_%d_%d_%.6f", site_id[0], site_id[1], site_id[2], site_id[3], cell_vol);
    if (cell_map.find(cell_key) == cell_map.end()) {
      cell_map[cell_key] = distinct_first_cell.size();
      distinct_first_cell.push_back(c);
      distinct_n_cells.push_back(0);
    }
    int cell_id = cell_map[cell_key];
    distinct_cell.push_back(cell_id);
    distinct_n_cells[cell_id]++;
  }
  double cr_mean = cr_sum / double(lattice.n_cells);
  double cr2_mean = cr2_sum / double(lattice.n_cells);
  double cr_stdev = sqrt((cr2_mean - cr_mean * cr_mean) / double(lattice.n_cells));
  printf("cell_vol: %.12f\n", cell_vol);
  printf("cr_mean: %.12f\n", cr_mean);
  printf("cr_stdev:  %.12e\n", cr_stdev);

  printf("cell degeneracies:\n");
  for (int i = 0; i < distinct_n_cells.size(); i++) {
    printf("%d %d\n", i, distinct_n_cells[i]);
  }

  // compute the initial error
  PrintError(lattice);
  double error_sum = CombinedError(lattice);
  double old_error = error_sum;

  double delta = 1.0e-5;
  double delta_sq = delta * delta;
  for (int n = 0; n < 1000; n++) {

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
      A(d1,d1) = (Fp + Fm - 2.0 * F0) / delta_sq;

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

        A(d1,d2) = 0.25 * (Fpp - Fpm - Fmp + Fmm) / delta_sq;

        // matrix is symmetric
        A(d2,d1) = A(d1,d2);
      }

      orbits[o1].dof[dof1] = base_value1;
      UpdateOrbit(lattice, o1);
    }

    int mu = 0;
    double lambda = 1.0;

    std::vector<double> old_dof(n_dof);
    for (int d = 0; d < n_dof; d++) {
      int o = dof_orbit[d];
      int dof = dof_index[d];
      old_dof[d] = orbits[o].dof[dof];
    }

    while (true) {

      // compute the solution
      Eigen::ConjugateGradient<Mat> cg;
      cg.compute(A);
      assert(cg.info() == Eigen::Success);
      Vec x = cg.solve(b);

      lambda = 1.0;
      while (true) {

        // apply the new positions the orbits
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
        // printf("%04d %02d %.2e %.12e\n", n, mu, lambda, error_sum);
        if (isnan(error_sum)) exit(1);

        lambda *= 0.5;
        if (lambda < 0.125) {
          break;
        }
      }

      if (error_sum < old_error) break;
      // printf("%04d %02d %.2e %.12e\n", n, mu, lambda, error_sum);
      if (isnan(error_sum)) exit(1);

      mu++;
      if (mu > 50) break;
      A += Mat::Identity(n_dof, n_dof);
    }

    // check the error
    double delta_err = (old_error - error_sum) / old_error;
    old_error = error_sum;
    printf("%04d %02d %.6f %.12e %.12e\n", n, mu, lambda, error_sum, delta_err);
    if (mu > 50) break;;
    if (lambda < 1.0) continue;
    if (delta_err < 1.0e-10 || error_sum < 1.0e-14) break;
  }

  // adjust octahedral centers
  for (int s = 0; s < lattice.n_sites; s++) {
    if (lattice.sites[s].nn != 6) continue;
    Vec4 r_mean = Vec4::Zero();
    for (int n = 0; n < 6; n++) {
      r_mean += lattice.r[lattice.sites[s].neighbors[n]];
    }
    lattice.r[s] = r_mean.normalized();
  }

  PrintError(lattice);

  char lattice_path[200];
  sprintf(lattice_path, "%s_uniform.dat", base_path);
  FILE* lattice_file = fopen(lattice_path, "w");
  assert(lattice_file != nullptr);
  lattice.WriteLattice(lattice_file);
  fclose(lattice_file);

  return 0;
}
