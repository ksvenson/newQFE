// ylm_mix.cc

// Computes the linear combinations of spherical harmonics that belong to
// a given irreducible representation of the symmetry groups I, O, and T.
// The procedure follows S.L. Altmann "On the symmetries of spherical
// harmonics".

// usage: ylm_mix l m group irrep
// l and m determine the spherical harmonic generator to use
// the group and irrep to use (valid values are listed below)

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// use long double floating pointer numbers. this is probably overkill, but
// this program is so fast it doesn't really matter
typedef long double Real;
typedef std::complex<Real> Complex;

const Real PI = 3.14159265358979323846264338328L;  // pi

// common euler angles
const Real P0_1 = 0.0L;  // zero
const Real P1_5 = 0.628318530717958647692528676656L;  // pi/5
const Real P2_5 = 1.25663706143591729538505735331L;  // 2pi/5
const Real P1_2 = 1.57079632679489661923132169164L;  // pi/2
const Real P3_5 = 1.88495559215387594307758602997L;  // 3pi/5
const Real P4_5 = 2.51327412287183459077011470662L;  // 4pi/5
const Real P1_1 = 3.14159265358979323846264338328L;  // pi

// tetrahedron character for irreps E1 and E2
const Complex T_C0(-0.5L, 0.866025403784438646763723170753L); // exp(2pi i/3)

// icosahedron beta angles
const Real I_B1 = 1.10714871779409050301706546018L;  // arccos(1/sqrt(5))
const Real I_B2 = 2.03444393579570273544557792310L;  // pi - arccos(1/sqrt(5))

const Real I_C0 = 1.61803398874989484820458683437L;  // golden ratio

struct Element {
  int conj_class;
  Real alpha;
  int beta;
  Real gamma;
};

inline Real factorial(int n) {
  return tgamma(n + 1);
}

void ylm_mix(const char* group, const char* irrep, int l, int m) {

  std::vector<Real> beta;
  std::vector<Complex> characters;
  std::vector<Element> elements;

  if (strcmp(group, "I") == 0) {

    // icosahedral rotation group

    if (strcmp(irrep, "A") == 0) {
      characters.push_back(1.0);  // conjugacy class E
      characters.push_back(1.0);  // conjugacy class 12C_5
      characters.push_back(1.0);  // conjugacy class 12C2_5
      characters.push_back(1.0);  // conjugacy class 20C_3
      characters.push_back(1.0);  // conjugacy class 15C_2

    } else if (strcmp(irrep, "T1") == 0) {
      characters.push_back(3.0);  // conjugacy class E
      characters.push_back(I_C0);  // conjugacy class 12C_5
      characters.push_back(1.0 - I_C0);  // conjugacy class 12C2_5
      characters.push_back(0.0);  // conjugacy class 20C_3
      characters.push_back(-1.0);  // conjugacy class 15C_2

    } else if (strcmp(irrep, "T2") == 0) {
      characters.push_back(3.0);  // conjugacy class E
      characters.push_back(1.0 - I_C0);  // conjugacy class 12C_5
      characters.push_back(I_C0);  // conjugacy class 12C2_5
      characters.push_back(0.0);  // conjugacy class 20C_3
      characters.push_back(-1.0);  // conjugacy class 15C_2

    } else if (strcmp(irrep, "G") == 0) {
      characters.push_back(4.0);  // conjugacy class E
      characters.push_back(-1.0);  // conjugacy class 12C_5
      characters.push_back(-1.0);  // conjugacy class 12C2_5
      characters.push_back(1.0);  // conjugacy class 20C_3
      characters.push_back(0.0);  // conjugacy class 15C_2

    } else if (strcmp(irrep, "H") == 0) {
      characters.push_back(5.0);  // conjugacy class E
      characters.push_back(0.0);  // conjugacy class 12C_5
      characters.push_back(0.0);  // conjugacy class 12C2_5
      characters.push_back(-1.0);  // conjugacy class 20C_3
      characters.push_back(1.0);  // conjugacy class 15C_2

    } else {
      fprintf(stderr, "irrep %s not implemented for group I\n", irrep);
      fprintf(stderr, "valid irreps: A, T1, T2, G, H\n");
      exit(0);
    }

    beta.push_back(P0_1);
    beta.push_back(I_B1);
    beta.push_back(I_B2);
    beta.push_back(P1_1);

    // conjugacy class E
    elements.push_back(Element{ 0, +P0_1, 0, +P0_1 });  // E

    // conjugacy class 12C_5
    elements.push_back(Element{ 1, +P0_1, 1, +P1_5 });  // C+5,2
    elements.push_back(Element{ 1, +P0_1, 1, -P1_5 });  // C+5,5
    elements.push_back(Element{ 1, +P2_5, 1, -P1_5 });  // C-5,4
    elements.push_back(Element{ 1, -P2_5, 1, +P1_5 });  // C+5,3
    elements.push_back(Element{ 1, +P2_5, 1, -P3_5 });  // C-5,1
    elements.push_back(Element{ 1, -P2_5, 1, +P3_5 });  // C+5,1
    elements.push_back(Element{ 1, +P4_5, 1, -P1_1 });  // C-5,2
    elements.push_back(Element{ 1, -P4_5, 1, +P1_1 });  // C-5,5
    elements.push_back(Element{ 1, +P4_5, 1, -P3_5 });  // C-5,3
    elements.push_back(Element{ 1, -P4_5, 1, +P3_5 });  // C+5,4
    elements.push_back(Element{ 1, +P2_5, 0, +P0_1 });  // C+5,6
    elements.push_back(Element{ 1, -P2_5, 0, +P0_1 });  // C-5,6

    // conjugacy class 12C_5^2
    elements.push_back(Element{ 2, +P1_5, 2, +P2_5 });  // C2+5,2
    elements.push_back(Element{ 2, -P1_5, 2, -P2_5 });  // C2+5,5
    elements.push_back(Element{ 2, +P1_5, 2, -P4_5 });  // C2-5,1
    elements.push_back(Element{ 2, -P1_5, 2, +P4_5 });  // C2+5,1
    elements.push_back(Element{ 2, +P3_5, 2, +P4_5 });  // C2-5,2
    elements.push_back(Element{ 2, -P3_5, 2, -P4_5 });  // C2-5,5
    elements.push_back(Element{ 2, +P3_5, 2, +P0_1 });  // C2-5,4
    elements.push_back(Element{ 2, -P3_5, 2, +P0_1 });  // C2+5,3
    elements.push_back(Element{ 2, +P1_1, 2, -P2_5 });  // C2-5,3
    elements.push_back(Element{ 2, -P1_1, 2, +P2_5 });  // C2+5,4
    elements.push_back(Element{ 2, +P4_5, 0, +P0_1 });  // C2+5,6
    elements.push_back(Element{ 2, -P4_5, 0, +P0_1 });  // C2-5,6

    // conjugacy class 20C_3
    elements.push_back(Element{ 3, +P0_1, 1, +P3_5 });  // C+3,b
    elements.push_back(Element{ 3, +P0_1, 1, -P3_5 });  // C-3,a
    elements.push_back(Element{ 3, +P2_5, 1, +P1_5 });  // C+3,c
    elements.push_back(Element{ 3, -P2_5, 1, -P1_5 });  // C+3,e
    elements.push_back(Element{ 3, +P2_5, 1, +P1_1 });  // C-3,b
    elements.push_back(Element{ 3, -P2_5, 1, +P1_1 });  // C+3,a
    elements.push_back(Element{ 3, +P4_5, 1, -P1_5 });  // C-3,f
    elements.push_back(Element{ 3, -P4_5, 1, +P1_5 });  // C+3,f
    elements.push_back(Element{ 3, +P4_5, 1, +P3_5 });  // C-3,c
    elements.push_back(Element{ 3, -P4_5, 1, -P3_5 });  // C-3,e
    elements.push_back(Element{ 3, +P1_5, 2, +P0_1 });  // C+3,d
    elements.push_back(Element{ 3, -P1_5, 2, +P0_1 });  // C+3,j
    elements.push_back(Element{ 3, +P1_5, 2, -P2_5 });  // C-3,g
    elements.push_back(Element{ 3, -P1_5, 2, +P2_5 });  // C+3,i
    elements.push_back(Element{ 3, +P3_5, 2, -P2_5 });  // C-3,h
    elements.push_back(Element{ 3, -P3_5, 2, +P2_5 });  // C+3,h
    elements.push_back(Element{ 3, +P3_5, 2, -P4_5 });  // C-3,i
    elements.push_back(Element{ 3, -P3_5, 2, +P4_5 });  // C+3,g
    elements.push_back(Element{ 3, +P1_1, 2, +P4_5 });  // C-3,d
    elements.push_back(Element{ 3, +P1_1, 2, -P4_5 });  // C-3,j

    // conjugacy class 15C_2
    elements.push_back(Element{ 4, +P0_1, 1, +P1_1 });  // C2,16
    elements.push_back(Element{ 4, +P2_5, 1, +P3_5 });  // C2,26
    elements.push_back(Element{ 4, -P2_5, 1, -P3_5 });  // C2,68
    elements.push_back(Element{ 4, +P4_5, 1, +P1_5 });  // C2,47
    elements.push_back(Element{ 4, -P4_5, 1, -P1_5 });  // C2,37
    elements.push_back(Element{ 4, +P1_5, 2, +P4_5 });  // C2,12
    elements.push_back(Element{ 4, -P1_5, 2, -P4_5 });  // C2,18
    elements.push_back(Element{ 4, +P3_5, 2, +P2_5 });  // C2,29
    elements.push_back(Element{ 4, -P3_5, 2, -P2_5 });  // C2,35
    elements.push_back(Element{ 4, +P1_1, 2, +P0_1 });  // C2,34
    elements.push_back(Element{ 4, +P0_1, 3, +P0_1 });  // C2,25
    elements.push_back(Element{ 4, +P1_5, 3, +P1_1 });  // C2,13
    elements.push_back(Element{ 4, -P1_5, 3, +P1_1 });  // C2,14
    elements.push_back(Element{ 4, +P2_5, 3, +P0_1 });  // C2,48
    elements.push_back(Element{ 4, -P2_5, 3, +P0_1 });  // C2,23

  } else if (strcmp(group, "O") == 0) {

    // octahedral rotation group

    if (strcmp(irrep, "A1") == 0) {
      characters.push_back(1.0);  // conjugacy class E
      characters.push_back(1.0);  // conjugacy class 8C_3
      characters.push_back(1.0);  // conjugacy class 6C'_2
      characters.push_back(1.0);  // conjugacy class 6C_4
      characters.push_back(1.0);  // conjugacy class 3C_2

    } else if (strcmp(irrep, "A2") == 0) {
      characters.push_back(1.0);  // conjugacy class E
      characters.push_back(1.0);  // conjugacy class 8C_3
      characters.push_back(-1.0);  // conjugacy class 6C'_2
      characters.push_back(-1.0);  // conjugacy class 6C_4
      characters.push_back(1.0);  // conjugacy class 3C_2

    } else if (strcmp(irrep, "E") == 0) {
      characters.push_back(2.0);  // conjugacy class E
      characters.push_back(-1.0);  // conjugacy class 8C_3
      characters.push_back(0.0);  // conjugacy class 6C'_2
      characters.push_back(0.0);  // conjugacy class 6C_4
      characters.push_back(2.0);  // conjugacy class 3C_2

    } else if (strcmp(irrep, "T1") == 0) {
      characters.push_back(3.0);  // conjugacy class E
      characters.push_back(0.0);  // conjugacy class 8C_3
      characters.push_back(-1.0);  // conjugacy class 6C'_2
      characters.push_back(1.0);  // conjugacy class 6C_4
      characters.push_back(-1.0);  // conjugacy class 3C_2

    } else if (strcmp(irrep, "T2") == 0) {
      characters.push_back(3.0);  // conjugacy class E
      characters.push_back(0.0);  // conjugacy class 8C_3
      characters.push_back(1.0);  // conjugacy class 6C'_2
      characters.push_back(-1.0);  // conjugacy class 6C_4
      characters.push_back(-1.0);  // conjugacy class 3C_2

    } else {
      fprintf(stderr, "irrep %s not implemented for group O\n", irrep);
      fprintf(stderr, "valid irreps: A1, A2, E, T1, T2\n");
      exit(0);
    }

    beta.push_back(P0_1);
    beta.push_back(P1_2);
    beta.push_back(P1_1);

    // conjugacy class E
    elements.push_back(Element{ 0, +P0_1, 0, +P0_1 });  // E

    // conjugacy class 8C_3
    elements.push_back(Element{ 1, +P0_1, 1, +P1_2 });  // C3a
    elements.push_back(Element{ 1, +P1_1, 1, -P1_2 });  // C3b
    elements.push_back(Element{ 1, +P0_1, 1, -P1_2 });  // C3c
    elements.push_back(Element{ 1, +P1_1, 1, +P1_2 });  // C3d
    elements.push_back(Element{ 1, +P1_2, 1, +P1_1 });  // C3a,2
    elements.push_back(Element{ 1, -P1_2, 1, +P0_1 });  // C3b,2
    elements.push_back(Element{ 1, -P1_2, 1, +P1_1 });  // C3c,2
    elements.push_back(Element{ 1, +P1_2, 1, +P0_1 });  // C3d,2

    // conjugacy class 6C'_2
    elements.push_back(Element{ 2, -P1_2, 2, +P0_1 });  // C2x+y
    elements.push_back(Element{ 2, +P1_2, 2, +P0_1 });  // C2x-y
    elements.push_back(Element{ 2, +P0_1, 1, +P1_1 });  // C2x+z
    elements.push_back(Element{ 2, +P1_1, 1, +P0_1 });  // C2x-z
    elements.push_back(Element{ 2, +P1_2, 1, +P1_2 });  // C2y+z
    elements.push_back(Element{ 2, -P1_2, 1, -P1_2 });  // C2y-z

    // conjugacy class 6C_4
    elements.push_back(Element{ 3, -P1_2, 1, +P1_2 });  // C4x
    elements.push_back(Element{ 3, +P0_1, 1, +P0_1 });  // C4y
    elements.push_back(Element{ 3, +P1_2, 0, +P0_1 });  // C4z
    elements.push_back(Element{ 3, +P1_2, 1, -P1_2 });  // C4x,3
    elements.push_back(Element{ 3, +P1_1, 1, +P1_1 });  // C4y,3
    elements.push_back(Element{ 3, -P1_2, 0, +P0_1 });  // C4z,3

    // conjugacy class 3C_2
    elements.push_back(Element{ 4, +P1_1, 2, +P0_1 });  // C2x
    elements.push_back(Element{ 4, +P0_1, 2, +P0_1 });  // C2y
    elements.push_back(Element{ 4, +P1_1, 0, +P0_1 });  // C2z

  } else if (strcmp(group, "T") == 0) {

    // tetrahedral rotation group

    if (strcmp(irrep, "A") == 0) {
      characters.push_back(1.0);  // conjugacy class E
      characters.push_back(1.0);  // conjugacy class 3C_2
      characters.push_back(1.0);  // conjugacy class 4C_3
      characters.push_back(1.0);  // conjugacy class 4C2_3

    } else if (strcmp(irrep, "E1") == 0) {
      characters.push_back(1.0);  // conjugacy class E
      characters.push_back(1.0);  // conjugacy class 3C_2
      characters.push_back(T_C0);  // conjugacy class 4C_3
      characters.push_back(conj(T_C0));  // conjugacy class 4C2_3

    } else if (strcmp(irrep, "E2") == 0) {
      characters.push_back(1.0);  // conjugacy class E
      characters.push_back(1.0);  // conjugacy class 3C_2
      characters.push_back(conj(T_C0));  // conjugacy class 4C_3
      characters.push_back(T_C0);  // conjugacy class 4C2_3

    } else if (strcmp(irrep, "T") == 0) {
      characters.push_back(3.0);  // conjugacy class E
      characters.push_back(-1.0);  // conjugacy class 3C_2
      characters.push_back(0.0);  // conjugacy class 4C+_3
      characters.push_back(0.0);  // conjugacy class 4C-_3

    } else {
      fprintf(stderr, "irrep %s not implemented for group T\n", irrep);
      fprintf(stderr, "valid irreps: A, E1, E2, T\n");
      exit(0);
    }

    beta.push_back(P0_1);
    beta.push_back(P1_2);
    beta.push_back(P1_1);

    // conjugacy class E
    elements.push_back(Element{ 0, +P0_1, 0, +P0_1 });  // E

    // conjugacy class 3C_2
    elements.push_back(Element{ 1, +P1_1, 2, +P0_1 });  // C2x
    elements.push_back(Element{ 1, +P0_1, 2, +P0_1 });  // C2y
    elements.push_back(Element{ 1, +P1_1, 0, +P0_1 });  // C2z

    // conjugacy class 4C+_3
    elements.push_back(Element{ 2, +P0_1, 1, +P1_2 });  // C+3,1
    elements.push_back(Element{ 2, +P1_1, 1, -P1_2 });  // C+3,2
    elements.push_back(Element{ 2, +P1_1, 1, +P1_2 });  // C+3,3
    elements.push_back(Element{ 2, +P0_1, 1, -P1_2 });  // C+3,4

    // conjugacy class 4C-_3
    elements.push_back(Element{ 3, +P1_2, 1, +P1_1 });  // C-3,1
    elements.push_back(Element{ 3, -P1_2, 1, +P0_1 });  // C-3,2
    elements.push_back(Element{ 3, +P1_2, 1, +P0_1 });  // C-3,3
    elements.push_back(Element{ 3, -P1_2, 1, +P1_1 });  // C-3,4

  } else {
    fprintf(stderr, "group %s not implemented\n", group);
    fprintf(stderr, "valid groups: I, O, T\n");
    exit(0);
  }

  int n_classes = characters.size();
  printf("n_classes: %d\n", n_classes);

  int n_beta = beta.size();
  printf("n_beta: %d\n", n_beta);

  int n_elements = elements.size();
  printf("n_elements: %d\n", n_elements);

  std::vector<std::vector<int>> beta_sets(n_beta);

  for (int i = 0; i < n_elements; i++) {
    beta_sets[elements[i].beta].push_back(i);
  }

  std::vector<Complex> result(2 * l + 1, 0.0L);

  // get coefficient of each Ylm
  for (int mp = -l; mp <= l; mp++) {
    // sum over beta sets (r in the paper)
    for (int b = 0; b < n_beta; b++) {

      // sum over conjugacy classes
      Complex class_sum = 0.0L;
      for (int s = 0; s < n_classes; s++) {

        // sum over group elements
        Complex element_sum = 0.0L;
        for (int t = 0; t < beta_sets[b].size(); t++) {
          int e = beta_sets[b][t];
          if (elements[e].conj_class != s) continue;

          Real mp_gamma = mp * elements[e].gamma;
          Real m_alpha = m * elements[e].alpha;
          Complex g = Complex(cos(mp_gamma), sin(mp_gamma));
          Complex a = Complex(cos(m_alpha), sin(m_alpha));
          element_sum += g * a;
        }

        Complex chi = characters[s];

        Real C = 1.0L;

        // flip sign if m is positive and odd
        if (m > 0 && m % 2) C *= -1.0L;

        // flip sign if m' is positive and odd
        if (mp > 0 && mp % 2) C *= -1.0L;

        class_sum += conj(chi) * C * element_sum;
      }

      // get script I

      int mu = std::min(l - mp, l + m);
      int nu = std::max(l - mp, l + m);
      int r_m = std::min(mu, 2 * l - nu);
      Real script_I = 0.0L;
      for (int r = 0; r <= r_m; r++) {

        // exponents for sin and cos factors
        int cos_p = nu - mu + 2 * r;
        int sin_p = 2 * l - cos_p;

        // sin and cos factors
        Real cos_beta;
        Real sin_beta;

        if (cos_p == 0) {
          // factor is 1 if exponent is zero
          cos_beta = 1.0L;
        } else if (beta[b] == PI) {
          // factor is zero if beta=pi
          cos_beta = 0.0L;
        } else {
          cos_beta = pow(cos(0.5L * beta[b]), cos_p);
        }

        if (sin_p == 0) {
          // factor is 1 if exponent is zero
          sin_beta = 1.0L;
        } else if (beta[b] == 0.0L) {
          // factor is zero if beta=0
          sin_beta = 0.0L;
        } else {
          sin_beta = pow(sin(0.5L * beta[b]), sin_p);
        }

        // this is the denominator of script_C in the paper
        Real script_C = factorial(nu - mu + r) * factorial(r) * \
            factorial(mu - r) * factorial(2 * l - nu - r);
        if ((mu - r) % 2) script_C *= -1.0L;

        script_I += cos_beta * sin_beta / script_C;
      }

      script_I *= factorial(l + abs(m)) * factorial(l - abs(mp));

      result[mp + l] += script_I * class_sum;
    }
  }

  Real norm_sum = 0.0L;
  for (int i = 0; i < result.size(); i++) {

    // convert to normalized spherical harmonics
    int abs_m = abs(i - l);
    result[i] *= sqrt(factorial(l + abs_m) / factorial(l - abs_m));
    norm_sum += norm(result[i]);
  }
  Real coeff = sqrt(norm_sum);
  printf("coeff: %.12Le\n", coeff);

  if (coeff < 1.0) {
    printf("no mixing\n");
    return;
  }

  // print normalized coefficients
  for (int i = 0; i < result.size(); i++) {

    result[i] /= coeff;
    if (abs(result[i]) < 1.0e-9) continue;

    printf("Y(%2d,%+3d) %+.18Lf %+.18Lf\n", l, i - l, \
        real(result[i]), imag(result[i]));
  }
}

int main(int argc, const char* argv[]) {

  // default parameters
  int l = 0;
  int m = 0;
  const char* group = "I";
  const char* irrep = "A";

  // get command line parameters
  if (argc > 1) l = atoi(argv[1]);
  if (argc > 2) m = atoi(argv[2]);
  if (argc > 3) group = argv[3];
  if (argc > 4) irrep = argv[4];

  printf("l: %d\n", l);
  printf("m: %d\n", m);
  printf("group: %s\n", group);
  printf("irrep: %s\n", irrep);

  assert(abs(m) <= l);

  ylm_mix(group, irrep, l, m);
  return 0;
}
