// s2_std_refine.cc

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <vector>

#include "s2.h"

// generate a "standard" triangular refinement of S^2

typedef Eigen::Vector3<double> Vec3;

enum OrbitType { V, E, F, VE, VF, EF, VEF };

struct Orbit {
  int id;
  OrbitType type;
  double dof[2];
  std::string name;
};

Orbit CreateOrbit(int k, int x, int y) {
  // barycentric coordinates
  std::vector<int> xi(3);
  xi[0] = x;
  xi[1] = y;
  xi[2] = k - x - y;
  std::sort(xi.begin(), xi.end());

  Orbit orbit;
  orbit.dof[0] = 0.0;
  orbit.dof[1] = 0.0;

  if (xi[2] == k) {
    orbit.type = OrbitType::V;
  } else if (xi[0] == 0 && xi[1] == xi[2]) {
    orbit.type = OrbitType::E;
  } else if (xi[0] == xi[1] && xi[1] == xi[2]) {
    orbit.type = OrbitType::F;
  } else if (xi[0] == 0) {
    orbit.type = OrbitType::VE;
    orbit.dof[0] = double(xi[2]) / double(k);
  } else if (xi[0] == xi[1]) {
    orbit.type = OrbitType::VF;
    orbit.dof[0] = double(xi[2]) / double(k);
  } else if (xi[1] == xi[2]) {
    orbit.type = OrbitType::EF;
    orbit.dof[0] = double(xi[2]) / double(k);
  } else {
    orbit.type = OrbitType::VEF;
    orbit.dof[0] = double(xi[2]) / double(k);
    orbit.dof[1] = double(xi[1]) / double(k);
  }

  // generate a hashable name
  char orbit_name[200];
  sprintf(orbit_name, "%d_%d_%d", xi[0], xi[1], xi[2]);
  orbit.name = std::string(orbit_name);

  return orbit;
}

// create a hashable name to identify the coordinates of each vertex
std::string VertexName(Vec3 &v) {
  // deal with negative zero
  double v_x = v.x();
  if (fabs(v_x) < 1.0e-12) v_x = 0.0;
  double v_y = v.y();
  if (fabs(v_y) < 1.0e-12) v_y = 0.0;
  double v_z = v.z();
  if (fabs(v_z) < 1.0e-12) v_z = 0.0;
  char vec_name[200];
  sprintf(vec_name, "%+.9f_%+.9f_%+.9f", v_x, v_y, v_z);
  return std::string(vec_name);
}

// create a hashable name to identify an x,y position in a tetrahedron
std::string XYName(int x, int y) {
  char xy_name[200];
  sprintf(xy_name, "%d_%d", x, y);
  return std::string(xy_name);
}

// create a hashable name to identify a distinct orbit in a tetrahedron
std::string OrbitName(int k, int x, int y) {
  // barycentric coordinates
  std::vector<int> xi(3);
  xi[0] = x;
  xi[1] = y;
  xi[2] = k - x - y;
  std::sort(xi.begin(), xi.end());
  char orbit_name[200];
  sprintf(orbit_name, "%d_%d_%d", xi[0], xi[1], xi[2]);
  return std::string(orbit_name);
}

int main(int argc, char *argv[]) {
  int k = 3;  // refinement level
  int q = 5;  // icosahedron

  if (argc > 1) k = atoi(argv[1]);
  if (argc > 2) q = atoi(argv[2]);

  // create a base polytope lattice
  QfeLatticeS2 base_lattice(q);

  // create an empty lattice to refine
  QfeLatticeS2 lattice(0);
  if (q == 4) {
    // k-refined octahedron has V = 4 k^2 + 2 vertices
    lattice.ResizeSites(4 * k * k + 2);
  } else {
    // k-refined icosahedron has V = 10 k^2 + 2 vertices
    lattice.ResizeSites(10 * k * k + 2);
  }

  std::map<std::string, int> coord_map;
  std::map<std::string, int> orbit_map;
  std::vector<Orbit> orbit_list;
  int s_next = 0;

  // loop over cells of base polytope
  for (int f = 0; f < base_lattice.n_faces; f++) {
    // vertices of base polytope cell
    Vec3 face_r[3];
    face_r[0] = base_lattice.r[base_lattice.faces[f].sites[0]];
    face_r[1] = base_lattice.r[base_lattice.faces[f].sites[1]];
    face_r[2] = base_lattice.r[base_lattice.faces[f].sites[2]];

    // unit vectors in the cell in xy basis
    Vec3 n_x = (face_r[1] - face_r[0]) / double(k);
    Vec3 n_y = (face_r[2] - face_r[0]) / double(k);

    std::map<std::string, int> xy_map;

    // loop over xy to find all sites and set their positions
    for (int x = 0; x <= k; x++) {
      for (int y = 0; y <= k; y++) {
        if ((x + y) > k) continue;

        // calculate the coordinates of this vertex
        Vec3 v = face_r[0] + x * n_x + y * n_y;
        std::string vec_name = VertexName(v);

        // check if the site already exists
        if (coord_map.find(vec_name) == coord_map.end()) {
          // check if the orbit already exists
          std::string orbit_name = OrbitName(k, x, y);
          if (orbit_map.find(orbit_name) == orbit_map.end()) {
            // create a new orbit
            Orbit orbit = CreateOrbit(k, x, y);
            orbit.id = orbit_list.size();
            orbit_map[orbit_name] = orbit.id;
            orbit_list.push_back(orbit);
            // printf("%s %d\n", orbit_name.c_str(), orbit.type);
          }

          // printf("%06d %.12f %.12f %.12f %.12f\n", s_next, v.x(), v.y(),
          // v.z(), v.w());

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
        xy_map[XYName(x, y)] = coord_map[vec_name];
      }
    }

    // add faces and links
    for (int x = 0; x <= k; x++) {
      for (int y = 0; y <= k; y++) {
        if ((x + y) > k) continue;

        if ((x + y) != k) {
          // "forward" triangle
          int s1 = xy_map[XYName(x, y)];
          int s2 = xy_map[XYName(x, y + 1)];
          int s3 = xy_map[XYName(x + 1, y)];
          lattice.AddFace(s1, s2, s3);
        }

        if ((x != 0) && (y != 0)) {
          // "backward" triangle
          int s1 = xy_map[XYName(x, y)];
          int s2 = xy_map[XYName(x, y - 1)];
          int s3 = xy_map[XYName(x - 1, y)];
          lattice.AddFace(s1, s2, s3);
        }
      }
    }
  }

  // project all sites onto a unit sphere
  lattice.Inflate();
  lattice.PrintCoordinates();

  // save the site positions and the graph to a file
  char grid_path[200];
  sprintf(grid_path, "s2_std/q%dk%d_std.dat", q, k);
  FILE *grid_file = fopen(grid_path, "w");
  assert(grid_file != nullptr);
  lattice.WriteLattice(grid_file);
  fclose(grid_file);

  // generate an orbit file
  char orbit_path[200];
  sprintf(orbit_path, "s2_std/q%dk%d_orbit.dat", q, k);
  FILE *orbit_file = fopen(orbit_path, "w");
  assert(orbit_file != nullptr);
  for (int o = 0; o < orbit_list.size(); o++) {
    Orbit *orbit = &orbit_list[o];
    fprintf(orbit_file, "%04d %02d %.16f %.16f\n", orbit->id, orbit->type,
            orbit->dof[0], orbit->dof[1]);
  }
  fclose(orbit_file);

  return 0;
}
