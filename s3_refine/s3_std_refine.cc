// s3_std_refine.cc

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <map>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "s3.h"

// generate a "standard" tetrahedral/octahedral refinement of S^3

typedef Eigen::Vector4<double> Vec4;

// create a hashable name to identify the coordinates of each vertex
std::string VertexName(Vec4& v) {
  // deal with negative zero
  double v_x = v.x(); if (fabs(v_x) < 1.0e-12) v_x = 0.0;
  double v_y = v.y(); if (fabs(v_y) < 1.0e-12) v_y = 0.0;
  double v_z = v.z(); if (fabs(v_z) < 1.0e-12) v_z = 0.0;
  double v_w = v.w(); if (fabs(v_w) < 1.0e-12) v_w = 0.0;
  char vec_name[200];
  sprintf(vec_name, "%+.9f_%+.9f_%+.9f_%+.9f", v_x, v_y, v_z, v_w);
  return std::string(vec_name);
}

// create a hashable name to identify an x,y,z position in a tetrahedron
std::string XYZName(int x, int y, int z) {
  char xyz_name[200];
  sprintf(xyz_name, "%d_%d_%d", x, y, z);
  return std::string(xyz_name);
}

// create a hashable name to identify a distinct orbit in a tetrahedron
std::string OrbitName(int k, int x, int y, int z) {
  std::vector<int> sorted(3);
  sorted[0] = x;
  sorted[1] = y;
  sorted[2] = z;
  for (int i = 0; i < 3; i++) {
    if (sorted[i] > k / 2) sorted[i] = k - sorted[i];
  }
  std::sort(sorted.begin(), sorted.end());
  char orbit_name[200];
  sprintf(orbit_name, "%d_%d_%d", sorted[0], sorted[1], sorted[2]);
  return std::string(orbit_name);
}

int main(int argc, char* argv[]) {

  int k = 3;  // refinement level
  int q = 5;  // 600-cell

  if (argc > 1) k = atoi(argv[1]);
  if (argc > 2) q = atoi(argv[3]);

  QfeLatticeS3 base_lattice(0);

  // read base polytope lattice
  char base_lattice_path[200];
  sprintf(base_lattice_path, "s3_%d.dat", q);
  FILE* base_lattice_file = fopen(base_lattice_path, "r");
  assert(base_lattice_file != nullptr);
  base_lattice.ReadLattice(base_lattice_file);
  fclose(base_lattice_file);

  // k-refined 600-cell has V = 40 * k * (5 * k^2 - 2) vertices
  // and C = 200 * k * (5 k^2 - 2) cells
  QfeLatticeS3 lattice(0);
  lattice.ResizeSites((200 * k * k - 80) * k);

  std::map<std::string, int> coord_map;
  std::map<std::string, int> orbit_map;
  int s_next = 0;
  int o_next = 0;

  // loop over cells of base polytope
  for (int c = 0; c < base_lattice.n_cells; c++) {

    // vertices of base polytope cell
    Vec4 cell_r[4];
    cell_r[0] = base_lattice.r[base_lattice.cells[c].sites[0]];
    cell_r[1] = base_lattice.r[base_lattice.cells[c].sites[1]];
    cell_r[2] = base_lattice.r[base_lattice.cells[c].sites[2]];
    cell_r[3] = base_lattice.r[base_lattice.cells[c].sites[3]];

    // unit vectors in the cell in xyz basis
    Vec4 n_x = -0.5 * (cell_r[0] + cell_r[1] - cell_r[2] - cell_r[3]) / double(k);
    Vec4 n_y = -0.5 * (cell_r[0] - cell_r[1] + cell_r[2] - cell_r[3]) / double(k);
    Vec4 n_z = -0.5 * (cell_r[0] - cell_r[1] - cell_r[2] + cell_r[3]) / double(k);

    std::map<std::string, int> xyz_map;

    // loop over xyz to find all sites and set their positions
    for (int x = 0; x <= k; x++) {
      for (int y = 0; y <= k; y++) {
        for (int z = 0; z <= k; z++) {
          if ((x + y + z) > (2 * k)) continue;
          if (x + y < z) continue;
          if (y + z < x) continue;
          if (z + x < y) continue;

          // calculate the coordinates of this vertex
          Vec4 v = cell_r[0] + x * n_x + y * n_y + z * n_z;
          std::string vec_name = VertexName(v);

          // check if the site already exists
          if (coord_map.find(vec_name) == coord_map.end()) {

            // check if the orbit already exists
            std::string orbit_name = OrbitName(k, x, y, z);
            if (orbit_map.find(orbit_name) == orbit_map.end()) {
              // create a new orbit
              orbit_map[orbit_name] = o_next;
              o_next++;
            }

            // printf("%06d %.12f %.12f %.12f %.12f\n", s_next, v.x(), v.y(), v.z(), v.w());

            // create a new site
            coord_map[vec_name] = s_next;
            lattice.r[s_next] = v;
            lattice.sites[s_next].nn = 0;
            lattice.sites[s_next].wt = 1.0;
            lattice.sites[s_next].id = orbit_map[orbit_name];
            s_next++;
            assert(s_next <= lattice.n_sites);
          }

          // generate the xyz name for this vertex
          xyz_map[XYZName(x, y, z)] = coord_map[vec_name];
        }
      }
    }

    // add cells, faces, and links
    for (int x = 0; x <= k; x++) {
      for (int y = 0; y <= k; y++) {
        for (int z = 0; z <= k; z++) {
          if ((x + y + z) > (2 * k)) continue;
          if (x + y < z) continue;
          if (y + z < x) continue;
          if (z + x < y) continue;

          if ((x + y + z) & 1) {
            // odd site -> octahedron center
            int s0 = xyz_map[XYZName(x, y, z)];
            int s1 = xyz_map[XYZName(x - 1, y, z)];
            int s2 = xyz_map[XYZName(x, y - 1, z)];
            int s3 = xyz_map[XYZName(x, y, z - 1)];
            int s4 = xyz_map[XYZName(x + 1, y, z)];
            int s5 = xyz_map[XYZName(x, y + 1, z)];
            int s6 = xyz_map[XYZName(x, y, z + 1)];
            lattice.AddCell(s0, s1, s2, s3);
            lattice.AddCell(s0, s1, s2, s6);
            lattice.AddCell(s0, s1, s5, s3);
            lattice.AddCell(s0, s4, s2, s3);
            lattice.AddCell(s0, s4, s5, s6);
            lattice.AddCell(s0, s4, s5, s3);
            lattice.AddCell(s0, s4, s2, s6);
            lattice.AddCell(s0, s1, s5, s6);
            continue;
          }

          if ((x + y + z) != (2 * k)) {
            // even site -> "forward" tetrahedron
            int s1 = xyz_map[XYZName(x, y, z)];
            int s2 = xyz_map[XYZName(x, y + 1, z + 1)];
            int s3 = xyz_map[XYZName(x + 1, y, z + 1)];
            int s4 = xyz_map[XYZName(x + 1, y + 1, z)];
            lattice.AddCell(s1, s2, s3, s4);
          }

          if ((x + y >= z + 2) && (y + z >= x + 2) && (z + x >= y + 2)) {
            // even site -> "backward" tetrahedron
            int s1 = xyz_map[XYZName(x, y, z)];
            int s2 = xyz_map[XYZName(x, y - 1, z - 1)];
            int s3 = xyz_map[XYZName(x - 1, y, z - 1)];
            int s4 = xyz_map[XYZName(x - 1, y - 1, z)];
            lattice.AddCell(s1, s2, s3, s4);
          }
        }
      }
    }
  }

  // project all sites onto a unit sphere
  lattice.Inflate();

  // save the site positions and the graph to a file
  char grid_path[200];
  sprintf(grid_path, "s3_std/q%dk%d_grid.dat", q, k);
  FILE* grid_file = fopen(grid_path, "w");
  assert(grid_file != nullptr);
  lattice.WriteLattice(grid_file);
  fclose(grid_file);

  return 0;
}
