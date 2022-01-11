// lattice.h

#pragma once

#include <algorithm>
#include <map>
#include <random>
#include <stack>
#include <string>
#include <vector>
#include "rng.h"

#define MAX_SITE_NEIGHBORS 12
#define MAX_LINK_FACES 3
#define MAX_FACE_EDGES 4

struct QfeSite {
  double wt;  // site weight
  int nn;  // number of nearest neighbors
  int links[MAX_SITE_NEIGHBORS];  // nearest neighbor links
  int neighbors[MAX_SITE_NEIGHBORS];  // nearest neighbor sites
  int id;
};

struct QfeLink {
  double wt;  // link weight
  int sites[2];  // sites attached by this link
  int n_faces;  // number of faces that include this link
  int faces[MAX_LINK_FACES];  // faces that include this link
};

struct QfeFace {
  double wt;  // face weight
  int n_edges;  // number of links (3 for triangle)
  int edges[MAX_FACE_EDGES];  // links around this face
  int sites[MAX_FACE_EDGES];  // sites around this face
};

class QfeLattice {

public:
  QfeLattice();
  void SeedRng(unsigned int seed);
  void InitRect(int Nx, int Ny, double wt_x, double wt_y);
  void InitTriangle(int N, double skew = 0.0);
  void InitTriangle(int N, double wt1, double wt2, double wt3);
  void InitTriangle(int Nx, int Ny, double wt1, double wt2, double wt3);
  virtual void ResizeSites(int n_sites);
  virtual void InterpolateSite(int s, int s_a, int s_b, int num, int den);
  int FindLink(int a, int b);
  int AddLink(int a, int b, double wt);
  int AddFace(int a, int b, int c);
  void UpdateDistinct();
  void Refine2D(int n_refine);
  void PrintSites();
  void PrintLinks();
  void PrintFaces();
  void CheckConnectivity();
  void CheckConsistency();

  int n_sites;
  int n_links;
  int n_faces;
  double vol;

  std::vector<QfeSite> sites;
  std::vector<QfeLink> links;
  std::vector<QfeFace> faces;

  // symmetrically distinct sites
  std::vector<int> distinct_n_sites;  // number of sites for each distinct id
  std::vector<int> distinct_first;  // representative site for distinct group
  int n_distinct;

  QfeRng rng;
};

QfeLattice::QfeLattice() {
  n_sites = 0;
  n_links = 0;
  n_faces = 0;
  vol = 0.0;
  n_distinct = 0;
}

/**
 * @brief Reset and seed the random number generator
 *
 * @param seed Random number generator seed value
 */

void QfeLattice::SeedRng(unsigned int seed) {
  rng = QfeRng(seed);
}

/**
 * @brief Create a flat, rectangular lattice with periodic boundary conditions
 *
 * @param Nx Lattice Size
 * @param Ny Lattice Size
 * @param wtx Link weights
 */

void QfeLattice::InitRect(int Nx, int Ny, double wt1, double wt2) {

  // create sites
  ResizeSites(Nx * Ny);
  vol = double(Nx * Ny);

  // set all site weights to 1.0
  for (int s = 0; s < n_sites; s++) {
    sites[s].wt = 1.0;
    sites[s].nn = 0;
    sites[s].id = 0;
  }

  // create links
  links.clear();

  for (int s = 0; s < n_sites; s++) {
    int x = s % Nx;
    int y = s / Nx;

    // add links in the "forward" direction (2 links per site)
    // each link will end up with 4 neighbors
    int xp1 = (x + 1) % Nx;
    int yp1 = (y + 1) % Ny;
    AddLink(s, xp1 + y * Nx, wt1);
    AddLink(s, x + yp1 * Nx, wt2);
  }
}

/**
 * @brief Creates a flat triangulated lattice with periodic boundary
 * conditions.
 *
 * @param N Lattice size
 * @param skew A value from 0 and 1 which determines how skewed the triangles
 * are. a value of 0 corresponds to equilateral triangles, and a value of 1
 * corresponds to right triangles, which is equivalent to a square lattice
 * because the diagonal link weights will be zero.
 */

void QfeLattice::InitTriangle(int N, double skew) {

  // if skew = 0.0, all weights are the same (equilateral triangles)
  // if skew = 1.0, the middle link weight is zero (right triangles)
  // average weight is 2/3
  double wt1 = (2.0 + skew) / 3.0;
  double wt2 = (2.0 - 2.0 * skew) / 3.0;
  double wt3 = wt1;

  InitTriangle(N, wt1, wt2, wt3);
}

/**
 * @brief Creates a flat triangulated lattice with periodic boundary
 * conditions.
 *
 * @param N Lattice size
 * @param wtx The weights given to links in the 3 directions of
 * the triangular lattice
 */

void QfeLattice::InitTriangle(int N, double wt1, double wt2, double wt3) {

  InitTriangle(N, N, wt1, wt2, wt3);
}

/**
 * @brief Creates a flat triangulated lattice with periodic boundary
 * conditions. Lattice dimensions can be asymmetrical.
 *
 * @param N Lattice size (x or short direction)
 * @param Nt Lattice size (y or long direction)
 * @param wtx The weights given to links in the 3 directions of
 * the triangular lattice
 */

void QfeLattice::InitTriangle(int Nx, int Ny, double wt1, double wt2, double wt3) {

  // create sites
  ResizeSites(Nx * Ny);
  vol = double(Nx * Ny);

  // set all site weights to 1.0
  for (int s = 0; s < n_sites; s++) {
    sites[s].wt = 1.0;
    sites[s].nn = 0;
    sites[s].id = 0;
  }

  // create links
  links.clear();

  for (int s = 0; s < n_sites; s++) {
    int x = s % Nx;
    int y = s / Nx;

    // add links in the "forward" direction (3 links per site)
    // each link will end up with 6 neighbors
    int xp1 = (x + 1) % Nx;
    int yp1 = (y + 1) % Ny;
    int xm1 = (x - 1 + Nx) % Nx;
    AddLink(s, xp1 + y * Nx, wt1);
    AddLink(s, x + yp1 * Nx, wt2);
    AddLink(s, xm1 + yp1 * Nx, wt3);
  }
}

/**
 * @brief Change the number of sites.
 */

void QfeLattice::ResizeSites(int n_sites) {
  this->n_sites = n_sites;
  sites.resize(n_sites);
}

/**
 * @brief Set the position of site @p s by interpolating between sites @p s_a
 * and @p s_b. The parameters @p num and @den define a fraction between 0 and 1
 * that determines how far from site @p s_a to put site @p s, with values of
 * 0 and 1 giving the coordinates of site a and site b, respectively.
 *
 * Subclasses can override this function to set the coordinates of the new
 * point.
 */

void QfeLattice::InterpolateSite(int s, int s_a, int s_b, int num, int den) {
  return;
}

/**
 * @brief Finds the link connecting sites @p a and @p b and returns the link
 * index. Returns -1 if no link exists.
 */

int QfeLattice::FindLink(int a, int b) {
  for (int n = 0; n < sites[a].nn; n++) {
    if (sites[a].neighbors[n] == b) return sites[a].links[n];
  }
  return -1;
}

/**
 * @brief Adds a link from site @p a to @p b with weight @p wt. Returns the
 * link index.
 */

int QfeLattice::AddLink(int a, int b, double wt) {

  int l = links.size();  // link index
  QfeLink link;
  link.wt = wt;
  link.sites[0] = a;
  link.sites[1] = b;
  link.n_faces = 0;

  // add site neighbors only if sites are dynamic
  if (a < n_sites) {
    int nn_a = sites[a].nn;
    sites[a].neighbors[nn_a] = b;
    sites[a].links[nn_a] = l;
    sites[a].nn++;
  }

  if (b < n_sites) {
    int nn_b = sites[b].nn;
    sites[b].neighbors[nn_b] = a;
    sites[b].links[nn_b] = l;
    sites[b].nn++;
  }

  links.push_back(link);
  n_links = links.size();
  return l;
}

/**
 * @brief Update the number of sites in each distinct group and find the
 * first site from each distinct group.
 */

void QfeLattice::UpdateDistinct() {
  distinct_n_sites.clear();
  distinct_first.clear();
  n_distinct = 0;

  for (int s = 0; s < n_sites; s++) {
    int id = sites[s].id;
    while (id >= n_distinct) {
      distinct_n_sites.push_back(0);
      distinct_first.push_back(0);
      n_distinct++;
    }

    if (distinct_n_sites[id] == 0) {
      // first site with this id
      distinct_first[id] = s;
    }
    distinct_n_sites[id]++;
    // printf("%04d %d\n", s, id);
  }
}

/**
 * @brief Add a triangular face with corner sites @p a, @p b, and @p c.
 * Returns the face index.
 */

int QfeLattice::AddFace(int a, int b, int c) {
  int f = faces.size();  // face index
  QfeFace face;
  face.wt = 1.0;
  face.n_edges = 3;
  int l;

  // link a to b
  face.sites[0] = a;
  l = FindLink(a, b);
  if (l == -1) l = AddLink(a, b, 1.0);
  face.edges[0] = l;
  links[l].faces[links[l].n_faces] = f;
  links[l].n_faces++;

  // link b to c
  face.sites[1] = b;
  l = FindLink(b, c);
  if (l == -1) l = AddLink(b, c, 1.0);
  face.edges[1] = l;
  links[l].faces[links[l].n_faces] = f;
  links[l].n_faces++;

  // link c to a
  face.sites[2] = c;
  l = FindLink(c, a);
  if (l == -1) l = AddLink(c, a, 1.0);
  face.edges[2] = l;
  links[l].faces[links[l].n_faces] = f;
  links[l].n_faces++;

  faces.push_back(face);
  n_faces = faces.size();
  return f;
}

/**
 * @brief Refine every face in a 2-dimensional lattice by splitting each link
 * into @p n_refine sublinks and partitioning the face appropriately.
 */

void QfeLattice::Refine2D(int n_refine) {

  if (n_refine < 2) return;

  // copy the old links and faces
  std::vector<QfeLink> old_links = links;
  std::vector<QfeFace> old_faces = faces;

  // remove all links and faces
  links.clear();
  n_links = 0;
  faces.clear();
  n_faces = 0;
  for (int s = 0; s < n_sites; s++) {
    sites[s].nn = 0;
  }

  int n_old_sites = n_sites;  // offset of new sites
  int n_new_sites = 0;

  // create n-1 new sites per edge
  n_new_sites += old_links.size() * (n_refine - 1);

  // create (n-1)(n-2)/2 interior sites per face
  n_new_sites += (old_faces.size() * (n_refine - 1) * (n_refine - 2)) / 2;

  ResizeSites(n_old_sites + n_new_sites);
  vol = double(n_sites);
  for (int s = n_old_sites; s < n_sites; s++) {
    sites[s].wt = 1.0;
    sites[s].nn = 0;
    sites[s].id = 0;
  }

  // map from an ordered pair of old sites to an array of new sites running
  // along the old edge between them
  std::map<std::string,std::vector<int>> edge_sites;

  int s = n_old_sites;
  for (int l = 0; l < old_links.size(); l++) {

    // corner sites for this edge
    int s_a = old_links[l].sites[0];
    int s_b = old_links[l].sites[1];

    std::vector<int> s_edge;
    s_edge.push_back(s_a);
    for (int n = 1; n < n_refine; n++) {
      InterpolateSite(s, s_a, s_b, n, n_refine);
      sites[s].id = (n_refine - abs(2 * n - n_refine)) / 2;
      s_edge.push_back(s);
      s++;
    }
    s_edge.push_back(s_b);

    // edge sites from a to b
    char key[50];
    sprintf(key, "%d_%d", s_a, s_b);
    edge_sites[key] = s_edge;

    // edge sites from b to a
    sprintf(key, "%d_%d", s_b, s_a);
    std::reverse(s_edge.begin(), s_edge.end());
    edge_sites[key] = s_edge;
  }

  // refine interior of old faces
  for (int f = 0; f < old_faces.size(); f++) {

    int s_corner[3];
    s_corner[0] = old_faces[f].sites[0];
    s_corner[1] = old_faces[f].sites[1];
    s_corner[2] = old_faces[f].sites[2];

    std::vector<int> e_outer[3];
    std::vector<int> e_inner[3];
    char key[50];
    sprintf(key, "%d_%d", s_corner[0], s_corner[1]);
    e_outer[0] = edge_sites[key];
    sprintf(key, "%d_%d", s_corner[1], s_corner[2]);
    e_outer[1] = edge_sites[key];
    sprintf(key, "%d_%d", s_corner[2], s_corner[0]);
    e_outer[2] = edge_sites[key];

    // distinct id of the first site in the first layer
    int layer_id = n_refine / 2 + 1;

    while (e_outer[0].size() >= 3) {
      int outer_size = e_outer[0].size();
      int inner_size = outer_size - 3;

      // total number of sites in inner loop
      int n_inner_loop = (inner_size - 1) * 3;
      if (inner_size == 0) {
        n_inner_loop = 0;
      } if (inner_size == 1) {
        // single center point
        n_inner_loop = 1;
      }

      // first site in the inner layer
      int s_inner = s;

      // add faces to connect inner and outer sites
      for (int e = 0; e < 3; e++) {

        int ep1 = (e + 1) % 3;  // next edge
        int em1 = (e + 2) % 3;  // previous edge

        // sites for interpolating
        int s_a = e_outer[em1][outer_size - 2];
        int s_b = e_outer[ep1][1];

        // corner triangle
        AddFace(e_outer[e][0], e_outer[e][1], s_a);

        e_inner[e].clear();

        for (int i = 0; i < inner_size; i++) {

          // connect to the previous inner site
          int sm1 = s;
          if (i == 0) {
            // when i=0, the previous site is on the previous outer edge
            sm1 = s_a;
          } else if (e == 2 && i == (inner_size - 1)) {
            // for the very last point, loop back around to the first point
            s = s_inner;
          } else if (inner_size != 1) {
            // if there is more than one inner site go to the next site
            s++;
          }

          // add two faces connecting this inner site
          AddFace(e_outer[e][i + 1], s, sm1);
          AddFace(e_outer[e][i + 1], e_outer[e][i + 2], s);

          e_inner[e].push_back(s);
          InterpolateSite(s, s_a, s_b, i + 1, inner_size + 1);
          sites[s].id = layer_id + (inner_size - 1 - abs(2 * i - (inner_size - 1))) / 2;
        }
      }

      // add center triangle for special cases
      if (inner_size == 0) {
        AddFace(e_outer[0][1], e_outer[1][1], e_outer[2][1]);
      } else if (inner_size == 2) {
        AddFace(e_inner[0][0], e_inner[1][0], e_inner[2][0]);
      }

      // go to the next layer on the interior of the face
      s = s_inner + n_inner_loop;
      e_outer[0] = e_inner[0];
      e_outer[1] = e_inner[1];
      e_outer[2] = e_inner[2];

      layer_id += (inner_size - 1) / 2 + 1;
      // printf("layer_id: %d\n", layer_id);
    }
  }
}

/**
 * @brief Print a list of sites with their weights and neighbors.
 */

void QfeLattice::PrintSites() {
  printf("\n*** sites ***\n");
  for (int s = 0; s < sites.size(); s++) {
    printf("%04d", s);
    printf(" %.12f", sites[s].wt);
    for (int n = 0; n < sites[s].nn; n++) {
      printf(" %04d", sites[s].neighbors[n]);
    }
    printf("\n");
  }
  printf("*************\n");
}

/**
 * @brief Print a list of links with their weights and attached sites.
 */

void QfeLattice::PrintLinks() {
  printf("\n*** links ***\n");
  for (int l = 0; l < links.size(); l++) {
    printf("%04d", l);
    printf(" %.12f", links[l].wt);
    printf(" %04d", links[l].sites[0]);
    printf(" %04d", links[l].sites[1]);
    for (int n = 0; n < links[l].n_faces; n++) {
      printf(" %04d", links[l].faces[n]);
    }
    printf("\n");
  }
  printf("*************\n");
}

void QfeLattice::PrintFaces() {
  printf("\n*** faces ***\n");
  for (int f = 0; f < faces.size(); f++) {
    printf("%04d", f);
    printf(" %.12f", faces[f].wt);
    for (int n = 0; n < faces[f].n_edges; n++) {
      printf(" %04d", faces[f].edges[n]);
    }
    printf("\n");
  }
  printf("*************\n");
}

/**
 * @brief Check that all lattice sites are connected.
 */

void QfeLattice::CheckConnectivity() {

  printf("\n*** connectivity check ***\n");
  printf("n_sites: %d\n", n_sites);

  // keep track of which sites are connected
  std::vector<bool> is_connected(sites.size());

  // create the stack
  std::stack<int> stack;

  // start with site 0
  stack.push(0);
  is_connected[0] = true;

  int n_connected = 0;

  while (stack.size() != 0) {
    n_connected++;
    int s = stack.top();
    stack.pop();
    QfeSite* site = &sites[s];

    for (int n = 0; n < site->nn; n++) {
      int s = site->neighbors[n];
      if (is_connected[s]) continue;
      is_connected[s] = true;
      stack.push(s);
    }
  }

  printf("connected sites: %d\n", n_connected);
  printf("disconnected sites: %d\n", int(sites.size()) - n_connected);
  printf("**************************\n");
}

/**
 * @brief Check that the neighbor lists match the link sites.
 */

void QfeLattice::CheckConsistency() {

  printf("\n*** consistency check ***\n");

  int n_inconsistent = 0;

  // make sure each site's neighbor table is consistent with its links
  for (int l = 0; l < links.size(); l++) {

    int s_a = links[l].sites[0];
    int s_b = links[l].sites[1];
    int n;

    // check 1st site
    for (n = 0; n < sites[s_a].nn; n++) {
      if (sites[s_a].links[n] == l) break;
    }

    if (n == sites[s_a].nn) {
      // link not found
      printf("link %04d not found in neighbor table for site %04d\n", l, s_a);
      n_inconsistent++;
    } else if (sites[s_a].neighbors[n] != s_b) {
      printf("site %04d neighbor %04d mismatch (link %04d, site %04d)\n", \
          s_a, sites[s_a].neighbors[n], l, s_b);
      n_inconsistent++;
    }

    // check 2nd site
    for (n = 0; n < sites[s_b].nn; n++) {
      if (sites[s_b].links[n] == l) break;
    }

    if (n == sites[s_b].nn) {
      // link not found
      printf("link %04d not found in neighbor table for site %04d\n", l, s_b);
      n_inconsistent++;
    } else if (sites[s_b].neighbors[n] != s_a) {
      printf("site %04d neighbor %04d mismatch (link %04d, site %04d)\n", \
          s_b, sites[s_b].neighbors[n], l, s_a);
      n_inconsistent++;
    }
  }
  printf("%d inconsistencies found\n", n_inconsistent);
  printf("*************************\n");
}
