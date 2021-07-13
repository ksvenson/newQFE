// s2_ct.cc

#include <getopt.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <future>
#include <map>
#include <vector>
#include <Eigen/Sparse>
#include "s2.h"
#include "timer.h"

#define PARALLEL

struct Result {
  int s;
  double M_inv_x;
  int iterations;
  double error;
  double solve_time;
};

void async_calc_ct(Result* r, Eigen::SparseMatrix<double>* M, int s) {

  // create the solver and analyze M
  Timer solve_timer;
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cg;
  cg.compute(*M);
  assert(cg.info() == Eigen::Success);

  // create a source and solve via conjugate gradient
  Eigen::VectorXd b = Eigen::VectorXd::Zero(M->rows());
  b(s) = 1.0;
  Eigen::VectorXd x = cg.solve(b);
  solve_timer.Stop();
  assert(cg.info() == Eigen::Success);

  r->s = s;
  r->M_inv_x = x(s);
  r->iterations = cg.iterations();
  r->error = cg.error();
  r->solve_time = solve_timer.Duration();
}

int main(int argc, char* argv[]) {

  Timer total_timer;
  Timer setup_timer;

  // default parameters
  int n_refine = 8;
  int q = 5;  // icosahedron
  double m0 = 1.8;
  bool use_loop = false;

  const struct option long_options[] = {
    { "n_refine", required_argument, 0, 'N' },
    { "q", required_argument, 0, 'q' },
    { "m0", required_argument, 0, 'm' },
    { "loop", no_argument, 0, 'l' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "Nqml";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'N': n_refine = atoi(optarg); break;
      case 'q': q = atoi(optarg); break;
      case 'm': m0 = std::stod(optarg); break;
      case 'l': use_loop = true; break;
      default: break;
    }
  }

  printf("n_refine: %d\n", n_refine);
  printf("q: %d\n", q);
  printf("m0: %.4f\n", m0);
  printf("using %s refinement\n", use_loop ? "loop" : "flat");

  QfeLatticeS2 lattice(q);
  if (use_loop) {
    int n_loop = log2(round(n_refine));
    printf("n_loop: %d\n", n_loop);
    if ((1 << n_loop) != n_refine) {
      fprintf(stderr, "loop refinement must be a power of 2\n");
      exit(0);
    }
    lattice.LoopRefine(n_loop);
  } else {
    lattice.Refine2D(n_refine);
  }
  lattice.Inflate();
  lattice.UpdateDistinct();
  lattice.UpdateWeights();

  std::vector<Eigen::Triplet<double>> M_elements;

  // add nearest-neighbor interaction terms
  for (int l = 0; l < lattice.n_links; l++) {
    QfeLink* link = &lattice.links[l];
    int a = link->sites[0];
    int b = link->sites[1];
    M_elements.push_back(Eigen::Triplet<double>(a, b, -link->wt));
    M_elements.push_back(Eigen::Triplet<double>(b, a, -link->wt));
  }

  // add self-interaction terms
  for (int s = 0; s < lattice.n_sites; s++) {
    QfeSite* site = &lattice.sites[s];
    double wt_sum = m0 * site->wt / double(lattice.n_sites);
    for (int n = 0; n < site->nn; n++) {
      int l = site->links[n];
      wt_sum += lattice.links[l].wt;
    }
    M_elements.push_back(Eigen::Triplet<double>(s, s, wt_sum));
  }

  Eigen::SparseMatrix<double> M(lattice.n_sites, lattice.n_sites);
  M.setFromTriplets(M_elements.begin(), M_elements.end());

#ifdef PARALLEL

  int max_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(max_threads);

  std::vector<Result> results(lattice.n_distinct);
  std::vector<double> M_inv(lattice.n_distinct);

  double ct_sum = 0.0;
  for (int i = 0; i < lattice.n_distinct; i += max_threads) {

    for (int t = 0; t < max_threads; t++) {
      int id = i + t;
      if (id >= lattice.n_distinct) break;
      int s = lattice.distinct_first[id];
      threads[t] = std::thread(async_calc_ct, &results[id], &M, s);
    }

    // find M inverse for each distinct site
    for (int t = 0; t < max_threads; t++) {

      int id = i + t;
      if (id >= lattice.n_distinct) break;
      threads[t].join();
      Result r = results[id];

      printf("\ndistinct site %i / %d\n", id, lattice.n_distinct);
      printf("M_inv_x: %.12e\n", r.M_inv_x);
      printf("iterations: %d\n", r.iterations);
      printf("error: %.12e\n", r.error);
      printf("solve time: %.12f\n", r.solve_time);

      // save M_inv in the map
      M_inv[id] = r.M_inv_x;
      double site_wt = lattice.sites[r.s].wt;
      ct_sum += M_inv[id] * double(lattice.distinct_n_sites[id]) * site_wt;
    }
  }

#else  // not PARALLEL

  // create the solver and analyze M
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cg;
  cg.compute(M);
  setup_timer.Stop();
  assert(cg.info() == Eigen::Success);
  printf("setup time: %.12f\n", setup_timer.Duration());

  // find M inverse for each distinct site
  double ct_sum = 0.0;
  std::vector<double> M_inv(lattice.n_distinct);
  for (int i = 0; i < lattice.n_distinct; i++) {
    int s = lattice.distinct_first[i];
    double site_wt = lattice.sites[s].wt;

    printf("\ndistinct site %i / %d\n", i, lattice.n_distinct);

    // create a source and solve via conjugate gradient
    Eigen::VectorXd b = Eigen::VectorXd::Zero(lattice.n_sites);
    b(s) = 1.0;
    Timer solve_timer;
    Eigen::VectorXd x = cg.solve(b);
    solve_timer.Stop();
    assert(cg.info() == Eigen::Success);

    // print result
    printf("M_inv_x: %.12e\n", x(s));
    printf("iterations: %ld\n", cg.iterations());
    printf("error: %.12e\n", cg.error());
    printf("solve time: %.12f\n", solve_timer.Duration());

    // save M_inv in the map
    M_inv[i] = x(s);
    ct_sum += M_inv[i] * double(lattice.distinct_n_sites[i]) * site_wt;
  }
#endif  // PARALLEL

  double ct_mean = ct_sum / double(lattice.n_sites);

  total_timer.Stop();
  printf("total time: %.12f\n", total_timer.Duration());

  // open an output file
  char path[50];
  if (use_loop) {
    sprintf(path, "ct/ct_%d_%d_loop.dat", q, n_refine);
  } else {
    sprintf(path, "ct/ct_%d_%d.dat", q, n_refine);
  }
  FILE* file = fopen(path, "w");
  assert(file != nullptr);

  for (int i = 0; i < lattice.n_distinct; i++) {
    int s = lattice.distinct_first[i];
    double site_wt = lattice.sites[s].wt;
    printf("%d %04d %.20f %.20e %d\n", i, s, site_wt, M_inv[i] - ct_mean, lattice.distinct_n_sites[i]);
    fprintf(file, "%+.20e\n", M_inv[i] - ct_mean);
  }
  fclose(file);

  return 0;
}
