// gen_grp_s3.cc

#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <Eigen/Dense>

// generate all 14,400 group elements of the symmetry group of the 600-cell

// https://physics.stackexchange.com/questions/351372/generate-all-elements-of-a-point-group-from-generating-set

// Grove, L. The characters of the hecatonicosahedroidal group.
// Journal für die reine und angewandte Mathematik (Crelles
// Journal) 1974, 160–169 (1974).

typedef long double Real;

// SO(4) group element struct
struct GrpElemSO4 {
  Eigen::Quaternion<Real> l;
  Eigen::Quaternion<Real> r;
  bool star;
  std::string name;
};

// multiply two group elements
GrpElemSO4 TimesGrpElem(GrpElemSO4& g1, GrpElemSO4& g2) {
  GrpElemSO4 result;
  if (g1.star) {
    // for quaternions, (ab)^* = b^* a^* because they're non-abelian
    result.l = g1.l * g2.r.conjugate();
    result.r = g2.l.conjugate() * g1.r;
  } else {
    result.l = g1.l * g2.l;
    result.r = g2.r * g1.r;
  }
  result.star = g1.star ^ g2.star;
  return result;
}

// create a hashable name to identify a group element
void SetGrpElemName(GrpElemSO4& g) {

  // generate a string to identify this unique group element
  char name[200];
  int lw = int(round(g.l.w() * 1.0e8));
  int lx = int(round(g.l.x() * 1.0e8));
  int ly = int(round(g.l.y() * 1.0e8));
  int lz = int(round(g.l.z() * 1.0e8));
  int rw = int(round(g.r.w() * 1.0e8));
  int rx = int(round(g.r.x() * 1.0e8));
  int ry = int(round(g.r.y() * 1.0e8));
  int rz = int(round(g.r.z() * 1.0e8));

  // sign convention, this avoids double counting group elements which
  // are equal up to a sign of l and r, i.e. (1,1) = (-1,-1).
  int s = 1;
  if (lw < 0) {
    s = -1;
  } else if (lw == 0 && lx < 0) {
    s = -1;
  } else if (lw == 0 && lx == 0 && ly < 0) {
    s = -1;
  } else if (lw == 0 && lx == 0 && ly == 0 && lz < 0) {
    s = -1;
  }
  sprintf(name, "%s%+010d%+010d%+010d%+010d%+010d%+010d%+010d%+010d",
    g.star ? "1" : "0", \
    s * lw, s * lx, s * ly, s * lz, \
    s * rw, s * rx, s * ry, s * rz);
  g.name = name;
}

int main(int argc, char* argv[]) {

  int q = 5;
  if (argc == 2) q = atoi(argv[1]);
  printf("q: %d\n", q);

  // array of generating elements
  std::vector<GrpElemSO4> G;

  if (q == 4) {
    // generators for 16-cell symmetries
    const Real sqrt1_2 = 0.70710678118654752440L;
    G.resize(6);
    G[0].l = {1.0, 0.0, 0.0, 0.0};
    G[0].r = {1.0, 0.0, 0.0, 0.0};
    G[0].star = false;
    G[1].l = {1.0, 0.0, 0.0, 0.0};
    G[1].r = {-1.0, 0.0, 0.0, 0.0};
    G[1].star = false;
    G[2].l = {1.0, 0.0, 0.0, 0.0};
    G[2].r = {0.0, 1.0, 0.0, 0.0};
    G[2].star = false;
    G[3].l = {sqrt1_2, sqrt1_2, 0.0, 0.0};
    G[3].r = {sqrt1_2, sqrt1_2, 0.0, 0.0};
    G[3].star = false;
    G[4].l = {0.5, 0.5, 0.5, 0.5};
    G[4].r = {-0.5, 0.5, 0.5, 0.5};
    G[4].star = false;
    G[5].l = {1.0, 0.0, 0.0, 0.0};
    G[5].r = {1.0, 0.0, 0.0, 0.0};
    G[5].star = true;

  } else if (q == 5) {

    // generators for 600-cell symmetries
    const Real alpha = 0.80901699437494742410L;
    const Real beta = 0.30901699437494742410L;
    G.resize(34);

    // K1
    G[0].l = {1.0, 0.0, 0.0, 0.0};
    G[0].r = {1.0, 0.0, 0.0, 0.0};
    G[0].star = false;
    // K2
    G[1].l = {1.0, 0.0, 0.0, 0.0};
    G[1].r = {-1.0, 0.0, 0.0, 0.0};
    G[1].star = false;
    // K3
    G[2].l = {0.0, 1.0, 0.0, 0.0};
    G[2].r = {0.0, 1.0, 0.0, 0.0};
    G[2].star = false;
    // K4
    G[3].l = {0.5, alpha, beta, 0.0};
    G[3].r = {0.5, alpha, beta, 0.0};
    G[3].star = false;
    // K5
    G[4].l = {0.5, alpha, beta, 0.0};
    G[4].r = {-0.5, -alpha, -beta, 0.0};
    G[4].star = false;
    // K6
    G[5].l = {beta, 0.5, alpha, 0.0};
    G[5].r = {beta, 0.5, alpha, 0.0};
    G[5].star = false;
    // K7
    G[6].l = {beta, 0.5, alpha, 0.0};
    G[6].r = {-beta, -0.5, -alpha, 0.0};
    G[6].star = false;
    // K8
    G[7].l = {alpha, beta, 0.5, 0.0};
    G[7].r = {alpha, beta, 0.5, 0.0};
    G[7].star = false;
    // K9
    G[8].l = {alpha, beta, 0.5, 0.0};
    G[8].r = {-alpha, -beta, -0.5, 0.0};
    G[8].star = false;
    // K10
    G[9].l = {1.0, 0.0, 0.0, 0.0};
    G[9].r = {0.0, 1.0, 0.0, 0.0};
    G[9].star = false;
    // K11
    G[10].l = {1.0, 0.0, 0.0, 0.0};
    G[10].r = {0.5, alpha, beta, 0.0};
    G[10].star = false;
    // K12
    G[11].l = {1.0, 0.0, 0.0, 0.0};
    G[11].r = {-0.5, -alpha, -beta, 0.0};
    G[11].star = false;
    // K13
    G[12].l = {1.0, 0.0, 0.0, 0.0};
    G[12].r = {beta, 0.5, alpha, 0.0};
    G[12].star = false;
    // K14
    G[13].l = {1.0, 0.0, 0.0, 0.0};
    G[13].r = {-beta, -0.5, -alpha, 0.0};
    G[13].star = false;
    // K15
    G[14].l = {1.0, 0.0, 0.0, 0.0};
    G[14].r = {alpha, beta, 0.5, 0.0};
    G[14].star = false;
    // K16
    G[15].l = {1.0, 0.0, 0.0, 0.0};
    G[15].r = {-alpha, -beta, -0.5, 0.0};
    G[15].star = false;
    // K17
    G[16].l = {0.0, 1.0, 0.0, 0.0};
    G[16].r = {0.5, alpha, beta, 0.0};
    G[16].star = false;
    // K18
    G[17].l = {0.0, 1.0, 0.0, 0.0};
    G[17].r = {beta, 0.5, alpha, 0.0};
    G[17].star = false;
    // K19
    G[18].l = {0.0, 1.0, 0.0, 0.0};
    G[18].r = {alpha, beta, 0.5, 0.0};
    G[18].star = false;
    // K20
    G[19].l = {0.5, alpha, beta, 0.0};
    G[19].r = {beta, 0.5, alpha, 0.0};
    G[19].star = false;
    // K21
    G[20].l = {0.5, alpha, beta, 0.0};
    G[20].r = {-beta, -0.5, -alpha, 0.0};
    G[20].star = false;
    // K22
    G[21].l = {0.5, alpha, beta, 0.0};
    G[21].r = {alpha, beta, 0.5, 0.0};
    G[21].star = false;
    // K23
    G[22].l = {0.5, alpha, beta, 0.0};
    G[22].r = {-alpha, -beta, -0.5, 0.0};
    G[22].star = false;
    // K24
    G[23].l = {beta, 0.5, alpha, 0.0};
    G[23].r = {alpha, beta, 0.5, 0.0};
    G[23].star = false;
    // K25
    G[24].l = {beta, 0.5, alpha, 0.0};
    G[24].r = {-alpha, -beta, -0.5, 0.0};
    G[24].star = false;
    // K26
    G[25].l = {1.0, 0.0, 0.0, 0.0};
    G[25].r = {1.0, 0.0, 0.0, 0.0};
    G[25].star = true;
    // K27
    G[26].l = {-1.0, 0.0, 0.0, 0.0};
    G[26].r = {1.0, 0.0, 0.0, 0.0};
    G[26].star = true;
    // K28
    G[27].l = {0.0, 1.0, 0.0, 0.0};
    G[27].r = {1.0, 0.0, 0.0, 0.0};
    G[27].star = true;
    // K29
    G[28].l = {0.5, alpha, beta, 0.0};
    G[28].r = {1.0, 0.0, 0.0, 0.0};
    G[28].star = true;
    // K30
    G[29].l = {-0.5, -alpha, -beta, 0.0};
    G[29].r = {1.0, 0.0, 0.0, 0.0};
    G[29].star = true;
    // K31
    G[30].l = {beta, 0.5, alpha, 0.0};  // typo in the paper
    G[30].r = {1.0, 0.0, 0.0, 0.0};
    G[30].star = true;
    // K32
    G[31].l = {-beta, -0.5, -alpha, 0.0};  // typo in the paper
    G[31].r = {1.0, 0.0, 0.0, 0.0};
    G[31].star = true;
    // K33
    G[32].l = {alpha, beta, 0.5, 0.0};
    G[32].r = {1.0, 0.0, 0.0, 0.0};
    G[32].star = true;
    // K34
    G[33].l = {-alpha, -beta, -0.5, 0.0};
    G[33].r = {1.0, 0.0, 0.0, 0.0};
    G[33].star = true;
  }

  // set the unique identifiers for the generators
  for (int g = 0; g < G.size(); g++) {
    SetGrpElemName(G[g]);
  }

  // start with the identity element only
  std::map<std::string, int> name_list;
  name_list[G[0].name] = 1;
  printf("%s\n", G[0].name.c_str());

  // list of all group elements
  std::vector<GrpElemSO4> L;
  L.push_back(G[0]);
  while (true) {

    // new elements
    std::vector<GrpElemSO4> new_ones;

    // multiply all current elements times all generators
    for (int g = 0; g < L.size(); g++) {
      for (int h = 0; h < G.size(); h++) {
        // generate a new group element
        GrpElemSO4 gh = TimesGrpElem(L[g], G[h]);
        SetGrpElemName(gh);

        // skip if it's not new
        if (name_list.find(gh.name) != name_list.end()) continue;

        // add it to the list of new ones
        new_ones.push_back(gh);
        name_list[gh.name] = 1;
        printf("%s\n", gh.name.c_str());
      }
    }

    // exit when no new element were generated
    if (!new_ones.size()) break;

    // add new elements to the list
    L.insert(L.end(), new_ones.begin(), new_ones.end());
  }

  // write group element data to file
  char path[50];
  sprintf(path, "grp_s3_q%d.dat", q);
  FILE* file = fopen(path, "w");
  assert(file != nullptr);
  for (int i = 0; i < L.size(); i++) {
    GrpElemSO4 g = L[i];
    fprintf(file, "%06d %+d ", i, g.star ? -1 : 1);
    fprintf(file, "%+.18Lf %+.18Lf %+.18Lf %+.18Lf ", \
      g.l.w(), g.l.x(), g.l.y(), g.l.z());
    fprintf(file, "%+.18Lf %+.18Lf %+.18Lf %+.18Lf\n", \
      g.r.w(), g.r.x(), g.r.y(), g.r.z());
  }
  fclose(file);

  printf("group order: %lu\n", L.size());

  return 0;
}
