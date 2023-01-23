// s3_std_uniform.cc

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/IterativeLinearSolvers>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>

#include "s3.h"
#include "timer.h"
#include "util.h"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Mat;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vec;

enum OrbitType { V, E, F, C, VE, VF, VC, EF, EC, FC, VEF, VEC, VFC, EFC, VEFC };

std::vector<OrbitType> orbit_type;

std::vector<int> distinct_n_cells;
std::vector<int> distinct_first_cell;
std::vector<bool> is_oct_cell;

std::set<int> primary_sites;
std::vector<std::vector<int>> deficit_s1;
std::vector<std::vector<int>> deficit_s2;
std::vector<std::vector<int>> deficit_s3;
std::vector<std::vector<int>> deficit_c;

double update_time = 0.0;
double cell_time = 0.0;
double deficit_time = 0.0;

void PrintOrbits(QfeLatticeS3& lattice) {
  for (int o = 0; o < lattice.n_distinct; o++) {
    Vec4 xi = lattice.orbit_xi[o];
    double xi1 = Chop(xi[0]);
    double xi2 = Chop(xi[1]);
    double xi3 = Chop(xi[2]);
    double xi4 = Chop(xi[3]);
    double xi_sum = xi[0] + xi[0] + xi[2] + xi[3];
    printf("%04d %02d", o, orbit_type[o]);
    printf(" %.6f %.6f %.6f %.6f %.6f", xi1, xi2, xi3, xi4, xi_sum);
    int s = lattice.distinct_first[o];
    if (lattice.sites[s].nn == 6 && o != 0) {
      printf(" oct\n");
    } else {
      printf(" tet\n");
    }
  }
}

void UpdateOrbit(QfeLatticeS3& lattice, int o) {
  Timer timer;
  Vec4 xi = lattice.orbit_xi[o];
  OrbitType type = orbit_type[o];
  switch (type) {
    case OrbitType::VE:
      xi[1] = 1.0 - xi[0];
      break;
    case OrbitType::VF:
      xi[1] = 0.5 - 0.5 * xi[0];
      xi[2] = xi[1];
      break;
    case OrbitType::VC:
      xi[1] = (1.0 - xi[0]) / 3.0;
      xi[2] = xi[1];
      break;
    case OrbitType::EF:
      xi[1] = xi[0];
      xi[2] = 1.0 - 2.0 * xi[0];
      break;
    case OrbitType::EC:
      xi[1] = xi[0];
      xi[2] = 0.5 - xi[0];
      break;
    case OrbitType::FC:
      xi[1] = xi[0];
      xi[2] = xi[0];
      break;
    case OrbitType::VEF:
      xi[2] = 1.0 - (xi[0] + xi[1]);
      break;
    case OrbitType::VEC:
      xi[2] = 0.5 - 0.5 * (xi[0] + xi[1]);
      break;
    case OrbitType::VFC:
      xi[2] = xi[1];
      break;
    case OrbitType::EFC:
      xi[1] = xi[0];
      break;
    case OrbitType::VEFC:
      break;
    default:
      printf("Invalid orbit %d\n", type);
      exit(1);
      break;
  }

  xi[3] = 1.0 - xi[0] - xi[1] - xi[2];
  lattice.orbit_xi[o] = xi;

  Vec4 v = lattice.CalcOrbitPos(o).normalized();

  // use the symmetry group data to calculate site coordinates
  std::set<int>::iterator it = primary_sites.begin();
  while (it != primary_sites.end()) {
    int s = *it++;
    if (lattice.site_orbit[s] != o) continue;
    int g = lattice.site_g[s];
    lattice.r[s] = (lattice.G[g] * v).normalized();
  }
  timer.Stop();
  update_time += timer.Duration();
}

double CellVolumeError(QfeLatticeS3& lattice) {
  Timer timer;

  double vol_sum = 0.0;
  double vol_sq_sum = 0.0;
  for (int i = 0; i < distinct_first_cell.size(); i++) {
    int c = distinct_first_cell[i];
    double vol = lattice.CellVolume(c);
    if (is_oct_cell[i]) vol *= 2.0;
    double orbit_size = double(distinct_n_cells[i]);
    vol_sum += vol * orbit_size;
    vol_sq_sum += vol * vol * orbit_size;
  }

  double vol_mean = vol_sum / double(lattice.n_cells);
  double vol_sq_mean = vol_sq_sum / double(lattice.n_cells);

  timer.Stop();
  cell_time += timer.Duration();

  return Chop(vol_sq_mean / (vol_mean * vol_mean) - 1.0);
}

double CircumradiusError(QfeLatticeS3& lattice) {
  double vol_sum = 0.0;
  double vol_sq_sum = 0.0;
  int n_total = 0;
  for (int i = 0; i < distinct_first_cell.size(); i++) {
    if (is_oct_cell[i]) continue;
    int c = distinct_first_cell[i];
    int s = lattice.cells[c].sites[0];
    Vec4 cc = lattice.CellCircumcenter(c);
    double cr = (cc - lattice.r[s]).norm();
    double vol = cr * cr * cr;
    double orbit_size = double(distinct_n_cells[i]);
    vol_sum += vol * orbit_size;
    vol_sq_sum += vol * vol * orbit_size;
    n_total += distinct_n_cells[i];
  }

  double vol_mean = vol_sum / double(n_total);
  double vol_sq_mean = vol_sq_sum / double(n_total);

  return Chop(vol_sq_mean / (vol_mean * vol_mean) - 1.0);

  // double vol_sum = 0.0;
  // std::vector<double> vol_list(distinct_first_cell.size());
  // int n_total = 0;
  // for (int i = 0; i < distinct_first_cell.size(); i++) {
  //   int c = distinct_first_cell[i];
  //   int s = lattice.cells[c].sites[0];
  //   Vec4 cc = lattice.CellCircumcenter(distinct_first_cell[i]);
  //   double cr = (cc - lattice.r[s]).norm();
  //   double vol = cr * cr * cr;
  //   vol_list[i] = vol;
  //   vol_sum += vol * double(distinct_n_cells[i]);
  //   n_total += distinct_n_cells[i];
  // }
  // double vol_mean = vol_sum / double(n_total);

  // double error_sum = 0.0;
  // for (int i = 0; i < distinct_first_cell.size(); i++) {
  //   if (is_oct_cell[i]) continue;
  //   double vol_err = 1.0 - vol_list[i] / vol_mean;
  //   error_sum += pow(vol_err, 2.0) * double(distinct_n_cells[i]);
  // }

  // return error_sum / double(n_total);
}

double DualVolumeError(QfeLatticeS3& lattice) {
  double vol_sum = 0.0;
  double vol_sq_sum = 0.0;

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
            CM(i, i) = 0.0;
            for (int j = i + 1; j < 5; j++) {
              CM(i, j) = (cell_r[i] - cell_r[j]).squaredNorm();
              CM(j, i) = CM(i, j);
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

          double x_ijk =
              2.0 * (CM(i, j) * CM(i, k) + CM(i, j) * CM(j, k) +
                     CM(i, k) * CM(j, k)) -
              (CM(i, j) * CM(i, j) + CM(i, k) * CM(i, k) + CM(j, k) * CM(j, k));
          double x_ijl =
              2.0 * (CM(i, j) * CM(i, l) + CM(i, j) * CM(j, l) +
                     CM(i, l) * CM(j, l)) -
              (CM(i, j) * CM(i, j) + CM(i, l) * CM(i, l) + CM(j, l) * CM(j, l));

          double A_tri_ijk = 0.25 * sqrt(x_ijk);
          double A_tri_ijl = 0.25 * sqrt(x_ijl);
          double dual_ijk = CM(i, k) + CM(j, k) - CM(i, j);
          double dual_ijl = CM(i, l) + CM(j, l) - CM(i, j);

          double h_ijk =
              sqrt(cell_cr_sq - CM(i, j) * CM(i, k) * CM(j, k) / x_ijk);
          double h_ijl =
              sqrt(cell_cr_sq - CM(i, j) * CM(i, l) * CM(j, l) / x_ijl);

          if (cell_xi(l) < 0.0) h_ijk *= -1.0;
          if (cell_xi(k) < 0.0) h_ijl *= -1.0;

          double wt =
              (dual_ijk * h_ijk / A_tri_ijk + dual_ijl * h_ijl / A_tri_ijl) /
              16.0;
          double vol = wt * CM(i, j) / 6.0;

          site_vol += vol;
        }
      }
    }
    double orbit_size = double(lattice.distinct_n_sites[o]);
    vol_sum += site_vol * orbit_size;
    vol_sq_sum += site_vol * site_vol * orbit_size;
    // if (o < 4) {
    //   printf("%d %.12f\n", o, site_vol);
    // }
  }
  double vol_mean = vol_sum / double(lattice.n_sites);
  double vol_sq_mean = vol_sq_sum / double(lattice.n_sites);

  return Chop(vol_sq_mean / (vol_mean * vol_mean) - 1.0);
}

double DeficitAngleError(QfeLatticeS3& lattice) {
  Timer timer;
  double deficit_sum = 0.0;
  double deficit_sq_sum = 0.0;
  for (int id = 0; id < lattice.n_distinct; id++) {
    double angle_sum = 0.0;
    int s = lattice.distinct_first[id];

    for (int n = 0; n < deficit_s1[id].size(); n++) {
      int s1 = deficit_s1[id][n];
      int s2 = deficit_s2[id][n];
      int s3 = deficit_s3[id][n];
      int c = deficit_c[id][n];

      double cell_vol = lattice.CellVolume(c);

      Vec4 v1 = (lattice.r[s1] - lattice.r[s]);
      Vec4 v2 = (lattice.r[s2] - lattice.r[s]);
      Vec4 v3 = (lattice.r[s3] - lattice.r[s]);

      double norm1 = v1.norm();
      double norm2 = v2.norm();
      double norm3 = v3.norm();

      double dot12 = v1.dot(v2);
      double dot23 = v2.dot(v3);
      double dot31 = v3.dot(v1);

      double den =
          norm1 * norm2 * norm3 + dot12 * norm3 + dot23 * norm1 + dot31 * norm2;
      if (AlmostEq(den, 0.0)) {
        angle_sum += M_PI;
      } else {
        double cell_angle = atan(6.0 * cell_vol / den);
        if (den < 0) {
          cell_angle += M_PI;
        }
        angle_sum += 2.0 * cell_angle;
      }

      // // calculate the solid angle using L'Huilier's theorem
      // Vec4 v1 = (lattice.r[s1] - lattice.r[s]).normalized();
      // Vec4 v2 = (lattice.r[s2] - lattice.r[s]).normalized();
      // Vec4 v3 = (lattice.r[s3] - lattice.r[s]).normalized();

      // double t12 = acos(v1.dot(v2));
      // double t23 = acos(v2.dot(v3));
      // double t31 = acos(v3.dot(v1));
      // double ts = 0.5 * (t12 + t23 + t31);
      // double t_prod = tan(0.5 * ts) * tan(0.5 * (ts - t12)) *
      //                 tan(0.5 * (ts - t23)) * tan(0.5 * (ts - t31));
      // angle_sum += 4.0 * atan(sqrt(t_prod));
    }

    double deficit_angle = 4.0 * M_PI - angle_sum;
    double orbit_size = double(lattice.distinct_n_sites[id]);
    deficit_sum += deficit_angle * orbit_size;
    deficit_sq_sum += deficit_angle * deficit_angle * orbit_size;

    // if (id < 4) {
    //   printf("%d %.12f\n", id, deficit_angle);
    // }
  }

  double deficit_mean = deficit_sum / double(lattice.n_sites);
  double deficit_sq_mean = deficit_sq_sum / double(lattice.n_sites);

  timer.Stop();
  deficit_time += timer.Duration();

  // printf("total_deficit: %.12f\n", deficit_sum);
  // exit(0);

  return Chop(deficit_sq_mean / (deficit_mean * deficit_mean) - 1.0);
}

double EdgeDeficit(QfeLatticeS3& lattice) {
  lattice.CalcFEMWeights();
  double a_lat = lattice.CalcLatticeSpacing();
  double deficit_sum = 0.0;
  // double check_sum = 0.0;
  int check_s = 0;
  for (int l = 0; l < lattice.n_links; l++) {
    int s_a = lattice.links[l].sites[0];
    int s_b = lattice.links[l].sites[1];
    int n_cells = 0;
    double edge_sum = 0.0;

    // find pairs of faces which for a tetrahedron
    for (int i1 = 0; i1 < lattice.links[l].n_faces; i1++) {
      int f1 = lattice.links[l].faces[i1];
      int s_c = lattice.faces[f1].sites[0];
      if (s_c == s_a || s_c == s_b) {
        s_c = lattice.faces[f1].sites[1];
        if (s_c == s_a || s_c == s_b) {
          s_c = lattice.faces[f1].sites[2];
        }
      }

      for (int i2 = i1 + 1; i2 < lattice.links[l].n_faces; i2++) {
        int f2 = lattice.links[l].faces[i2];
        int s_d = lattice.faces[f2].sites[0];
        if (s_d == s_a || s_d == s_b) {
          s_d = lattice.faces[f2].sites[1];
          if (s_d == s_a || s_d == s_b) {
            s_d = lattice.faces[f2].sites[2];
          }
        }

        if (lattice.FindLink(s_c, s_d) == -1) continue;
        n_cells++;

        Vec4 n1 = (lattice.r[s_b] - lattice.r[s_a]).normalized();
        Vec4 n2 = (lattice.r[s_c] - lattice.r[s_a]).normalized();
        Vec4 n3 = (lattice.r[s_d] - lattice.r[s_a]).normalized();

        double cos23 = n2.dot(n3);
        double cos12 = n1.dot(n2);
        double cos13 = n1.dot(n3);
        double sin12 = sqrt(1.0 - cos12 * cos12);
        double sin13 = sqrt(1.0 - cos13 * cos13);

        double cos_phi = (cos23 - cos12 * cos13) / (sin12 * sin13);
        edge_sum += acos(cos_phi);
      }
    }
    double deficit_angle = 2.0 * M_PI - edge_sum;
    double edge_dual_area = lattice.links[l].wt * a_lat * lattice.EdgeLength(l);
    deficit_sum += deficit_angle;
    if (s_a == check_s || s_b == check_s) {
      // check_sum += 2.0 * M_PI - edge_sum;
      printf("%d %d %.12f\n", s_a, s_b, deficit_angle / edge_dual_area);
    }
  }
  printf("a_lat: %.12f\n", a_lat);
  printf("edge_deficit_sum: %.12f\n", deficit_sum);
  exit(0);
}

double CombinedError(QfeLatticeS3& lattice) {
  // return CellVolumeError(lattice);
  // return CircumradiusError(lattice);
  // return DualVolumeError(lattice);
  // return CellVolumeError(lattice) + CircumradiusError(lattice);
  // return CellVolumeError(lattice) + CircumradiusError(lattice) +
  //        DualVolumeError(lattice);
  return CellVolumeError(lattice) + DualVolumeError(lattice);
}

void PrintError(QfeLatticeS3& lattice) {
  double cell_err = CellVolumeError(lattice);
  double cr_err = CircumradiusError(lattice);
  double dual_err = DualVolumeError(lattice);
  double deficit_err = DeficitAngleError(lattice);
  double sum_err = cell_err + cr_err + dual_err;
  // EdgeDeficit(lattice);
  printf("cell_err: %.12e\n", cell_err);
  printf("cr_err:   %.12e\n", cr_err);
  printf("dual_err: %.12e\n", dual_err);
  printf("deficit_err: %.12e\n", deficit_err);
  printf("sum_err:  %.12e\n", sum_err);
}

int main(int argc, const char* argv[]) {
  int q = 5;
  int k = 2;

  if (argc > 1) q = atoi(argv[1]);
  if (argc > 2) k = atoi(argv[2]);

  QfeLatticeS3 lattice(q, k);
  lattice.UpdateDistinct();

  // generate the lists of sites for calculating deficit angles
  deficit_s1.resize(lattice.n_distinct);
  deficit_s2.resize(lattice.n_distinct);
  deficit_s3.resize(lattice.n_distinct);
  deficit_c.resize(lattice.n_distinct);
  for (int id = 0; id < lattice.n_distinct; id++) {
    int s = lattice.distinct_first[id];
    for (int n1 = 0; n1 < lattice.sites[s].nn; n1++) {
      int s1 = lattice.sites[s].neighbors[n1];
      for (int n2 = n1 + 1; n2 < lattice.sites[s].nn; n2++) {
        int s2 = lattice.sites[s].neighbors[n2];
        for (int n3 = n2 + 1; n3 < lattice.sites[s].nn; n3++) {
          int s3 = lattice.sites[s].neighbors[n3];

          int c = lattice.FindCell(s, s1, s2, s3);
          if (c == -1) continue;
          deficit_s1[id].push_back(s1);
          deficit_s2[id].push_back(s2);
          deficit_s3[id].push_back(s3);
          deficit_c[id].push_back(c);
        }
      }
    }
  }

  std::vector<int> dof_orbit;  // orbit number for each degree of freedom
  std::vector<int> dof_index;  // coordinate index within each orbit
  orbit_type.resize(lattice.n_distinct);  // type of each orbit
  for (int o = 0; o < lattice.n_distinct; o++) {
    Vec4 xi = lattice.orbit_xi[o];

    // determine the orbit type
    OrbitType type;
    if (AlmostEq(xi[0], 1.0)) {
      type = OrbitType::V;
    } else if (AlmostEq(xi[0], 0.5) && AlmostEq(xi[1], 0.5)) {
      type = OrbitType::E;
    } else if (AlmostEq(xi[0], xi[1]) && AlmostEq(xi[1], xi[2]) &&
               AlmostEq(xi[3], 0.0)) {
      type = OrbitType::F;
    } else if (AlmostEq(xi[0], xi[1]) && AlmostEq(xi[0], xi[2]) &&
               AlmostEq(xi[0], xi[3])) {
      type = OrbitType::C;
    } else if (AlmostEq(xi[3], 0.0) && AlmostEq(xi[2], 0.0)) {
      type = OrbitType::VE;
    } else if (AlmostEq(xi[3], 0.0) && AlmostEq(xi[2], xi[1])) {
      type = OrbitType::VF;
    } else if (AlmostEq(xi[3], xi[2]) && AlmostEq(xi[2], xi[1])) {
      type = OrbitType::VC;
    } else if (AlmostEq(xi[3], 0.0) && AlmostEq(xi[1], xi[0])) {
      type = OrbitType::EF;
    } else if (AlmostEq(xi[3], xi[2]) && AlmostEq(xi[1], xi[0])) {
      type = OrbitType::EC;
    } else if (AlmostEq(xi[0], xi[1]) && AlmostEq(xi[1], xi[2])) {
      type = OrbitType::FC;
    } else if (AlmostEq(xi[3], 0.0)) {
      type = OrbitType::VEF;
    } else if (AlmostEq(xi[3], xi[2])) {
      type = OrbitType::VEC;
    } else if (AlmostEq(xi[2], xi[1])) {
      type = OrbitType::VFC;
    } else if (AlmostEq(xi[1], xi[0])) {
      type = OrbitType::EFC;
    } else {
      type = OrbitType::VEFC;
    }

    orbit_type[o] = type;
    // printf("%04d %d\n", o, type);

    // count the number of degrees of freedom in this orbit
    if (type >= OrbitType::VE) {
      dof_orbit.push_back(o);
      dof_index.push_back(0);
    }

    if (type >= OrbitType::VEF && type != OrbitType::EFC) {
      dof_orbit.push_back(o);
      dof_index.push_back(1);
    }

    if (type >= OrbitType::EFC) {
      dof_orbit.push_back(o);
      dof_index.push_back(2);
    }
  }
  int n_dof = dof_orbit.size();
  printf("n_dof: %d\n", n_dof);

  // find distinct cells
  for (int c = 0; c < lattice.n_cells; c++) {
    int id = lattice.cell_orbit[c];
    while (id >= distinct_n_cells.size()) {
      distinct_n_cells.push_back(0);
      distinct_first_cell.push_back(0);
      is_oct_cell.push_back(false);
    }

    if (distinct_n_cells[id] == 0) {
      // first site with this id
      distinct_first_cell[id] = c;

      // check if this cell is part of an octahedron
      for (int i = 0; i < 4; i++) {
        int s = lattice.cells[c].sites[i];
        if (lattice.sites[s].nn == 6 && lattice.site_orbit[s] != 0) {
          is_oct_cell[id] = true;
          break;
        }
      }
    }

    distinct_n_cells[id]++;
  }

  // generate the set of primary sites
  for (int id = 0; id < lattice.n_distinct; id++) {
    int s = lattice.distinct_first[id];
    primary_sites.insert(s);
    for (int n = 0; n < lattice.sites[s].nn; n++) {
      int s1 = lattice.sites[s].neighbors[n];
      primary_sites.insert(s1);
    }
  }

  for (int i = 0; i < distinct_first_cell.size(); i++) {
    int c = distinct_first_cell[i];
    primary_sites.insert(lattice.cells[c].sites[0]);
    primary_sites.insert(lattice.cells[c].sites[1]);
    primary_sites.insert(lattice.cells[c].sites[2]);
    primary_sites.insert(lattice.cells[c].sites[3]);
  }
  printf("# of primary sites: %lu\n", primary_sites.size());

  // printf("cell degeneracies:\n");
  // for (int i = 0; i < distinct_n_cells.size(); i++) {
  //   int c = distinct_first_cell[i];
  //   printf("%d %d %.6e %s\n", i, distinct_n_cells[i], lattice.CellVolume(c),
  //          is_oct_cell[i] ? "oct" : "tet");
  // }
  // PrintOrbits(lattice);

  // compute the initial error
  PrintError(lattice);
  double error_sum = CombinedError(lattice);
  double old_error = error_sum;

  double delta = 1.0e-5;
  double delta_sq = delta * delta;
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
          if (lattice.site_orbit[s_n] == o2) {
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
    // PrintOrbits(lattice);

    int mu = 0;
    double lambda = 1.0;

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

      lambda = 1.0;
      while (true) {
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

        lambda *= 0.5;
        if (lambda < 0.125) {
          break;
        }
      }

      if (error_sum < old_error) break;
      // printf("%04d %02d %.2e %.12e\n", n, mu, lambda, error_sum);
      assert(!isnan(error_sum));

      mu++;
      if (mu > 50) break;
      A += Mat::Identity(n_dof, n_dof);
    }

    // check the error
    double delta_err = (old_error - error_sum) / old_error;
    old_error = error_sum;
    printf("%04d %02d %.4f %.12e %.12e\n", n, mu, lambda, error_sum, delta_err);
    if (mu > 50) break;
    // if (lambda < 1.0) continue;
    if (delta_err < 1.0e-10 || error_sum < 1.0e-14) break;
  }

  // PrintOrbits(lattice);
  PrintError(lattice);

  std::string orbit_path = string_format("orbit_q%dk%d.dat", q, k);
  printf("writing orbit file: %s\n", orbit_path.c_str());
  FILE* orbit_file = fopen(orbit_path.c_str(), "w");
  lattice.WriteOrbits(orbit_file);
  fclose(orbit_file);

  printf("update_time: %.6f\n", update_time);
  printf("cell_time: %.6f\n", cell_time);
  printf("deficit_time: %.6f\n", deficit_time);

  return 0;
}
