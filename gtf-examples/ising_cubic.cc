// ising_cubic.cc

#include <getopt.h>
//#include <cassert>
//#include <cmath>
//#include <cstdio>
#include <string>
#include <vector>
//#include <Eigen/Dense>
#include "ising-GTF.h"
//#include "statistics.h"
#include <iostream>
#include <fstream>
//#include <filesystem>
#include <sys/stat.h>

int main(int argc, char* argv[]) {

  int Nx = 4 ;
  int Ny = 4 ;
  int Nz = 4 ;

  double beta = 0.1020707 ;

  unsigned int seed = 1234u; // rng seed

  std::vector<double> sc = {0., 0., 0.} ;
  std::vector<double> fcc = {1., 1., 1., 1.06, 1.06, 1.06} ;
  std::vector<double> bcc = {0., 0., 0., 0.} ;

  int n_therm = 2000;
  int n_traj = 50000;
  int n_wolff = 3;

  std::string data_base = "ising_cubic";

  // Command-line Options
  const struct option long_options[] = {
    { "nx",        required_argument, 0, 'X' },
    { "ny",        required_argument, 0, 'Y' },
    { "nz",        required_argument, 0, 'Z' },
    { "seed",      required_argument, 0, 'S' },
    { "beta",      required_argument, 0, 'B' },
    { "k01",       required_argument, 0, 'C' },
    { "k02",       required_argument, 0, 'D' },
    { "k03",       required_argument, 0, 'E' },
    { "k04",       required_argument, 0, 'F' },
    { "k05",       required_argument, 0, 'G' },
    { "k06",       required_argument, 0, 'H' },
    { "k07",       required_argument, 0, 'I' },
    { "k08",       required_argument, 0, 'J' },
    { "k09",       required_argument, 0, 'K' },
    { "k10",       required_argument, 0, 'L' },
    { "k11",       required_argument, 0, 'M' },
    { "k12",       required_argument, 0, 'N' },
    { "k13",       required_argument, 0, 'O' },
    { "n_therm",   required_argument, 0, 'h' },
    { "n_traj",    required_argument, 0, 't' },
    { "n_wolff",   required_argument, 0, 'w' },
    { "data_base", required_argument, 0, 'd' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "X:Y:Z:S:B:C:D:E:F:G:H:I:J:K:L:M:N:O:h:t:w:d:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'X': Nx = atoi(optarg); break;
      case 'Y': Ny = atoi(optarg); break;
      case 'Z': Nz = atoi(optarg); break;
      case 'S': seed = atol(optarg); break;
      case 'B': beta = std::stod(optarg); break;
      case 'C': sc[0] = std::stod(optarg); break;
      case 'D': sc[1] = std::stod(optarg); break;
      case 'E': sc[2] = std::stod(optarg); break;
      case 'F': fcc[0] = std::stod(optarg); break;
      case 'G': fcc[1] = std::stod(optarg); break;
      case 'H': fcc[2] = std::stod(optarg); break;
      case 'I': fcc[3] = std::stod(optarg); break;
      case 'J': fcc[4] = std::stod(optarg); break;
      case 'K': fcc[5] = std::stod(optarg); break;
      case 'L': bcc[0] = std::stod(optarg); break;
      case 'M': bcc[1] = std::stod(optarg); break;
      case 'N': bcc[2] = std::stod(optarg); break;
      case 'O': bcc[3] = std::stod(optarg); break;
      case 'h': n_therm = atoi(optarg); break;
      case 't': n_traj = atoi(optarg); break;
      case 'w': n_wolff = atoi(optarg); break;
      case 'd': data_base = optarg; break;
      default: break;
    }
  }

  if (mkdir(data_base.c_str(), 02775) && errno != EEXIST) {
    perror(data_base.c_str()) ;
  }

  char tmp_buffer[66] ;

  std::snprintf(tmp_buffer, 66,
    "%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f",
    sc[0], sc[1], sc[2],
    fcc[0], fcc[1], fcc[2], fcc[3], fcc[4], fcc[5],
    bcc[0], bcc[1], bcc[2], bcc[3]) ;

  std::string buffer = tmp_buffer ;

  std::string data_dir = data_base + "/" + buffer ;

  // std::cout << data_dir << std::endl ;

  if (mkdir(data_dir.c_str(), 02775) && errno != EEXIST) {
    perror(data_dir.c_str()) ;
  }

  // exit(0) ;
   
  // std::filesystem::create_directory(data_dir) ;

  std::string param_fname = data_dir + "/"
    + std::to_string(Nx) + "_"
    + std::to_string(Ny) + "_"
    + std::to_string(Nz) + "_"
    + std::to_string(beta) + "_"
    + std::to_string(seed) + ".param" ;

  std::ofstream param_file ;

  param_file.open(param_fname) ;

  param_file << "X " << Nx << std::endl ;
  param_file << "Y " << Ny << std::endl ;
  param_file << "Z " << Nz << std::endl ;
  param_file << "S " << seed << std::endl ;
  param_file << "B " << beta << std::endl ;
  param_file << "h " << n_therm << std::endl ;
  param_file << "t " << n_traj << std::endl ;
  param_file << "w " << n_wolff << std::endl ;

  // std::cout << Nx << " " << Ny << " " << Nz << " " << seed << " " << beta << " " << n_therm
  //   << " " << n_traj << " " << n_wolff << " " << data_dir << std::endl ;

  std::vector<int> L = {Nx, Ny, Nz} ;

  // initialize the lattice
  QfeLattice lattice;
  lattice.SeedRng(seed);
  lattice.InitCubic(L, sc, fcc, bcc) ;

  int conn_sites = 0 ;
  for (int s=0; s < lattice.n_sites; s++) {
    if (lattice.sites[s].nn >= 1) { conn_sites++ ; }
  }

  // param_file << "link.wt" ;
  // for (int no = 0; no < lattice.sites[0].nn; no++) {
  //   param_file << " " << lattice.links[lattice.sites[0].links[no]].wt ;
  // }
  // param_file << std::endl ;

  for (int dir = 1; dir <= 13; dir++) {
    bool printed = false ;
    param_file << "link" << dir << " " ;
    for (int no = 0; no < lattice.sites[0].nn; no++) {
      if (lattice.sites[0].linkdirs[no] == dir && !printed ) {
        param_file << lattice.links[lattice.sites[0].links[no]].wt << std::endl ;
        printed = true ;
      }
    }
    if (!printed) {
      param_file << "0.0" << std::endl ;
    }
  }

  param_file << "n_sites " << lattice.n_sites << std::endl ;
  param_file << "n_links " << lattice.n_links << std::endl ;
  param_file << "conn_sites " << conn_sites << std::endl ;

  param_file.close() ;

  // std::cout << "vol " << lattice.vol << " n_sites " << lattice.n_sites << " n_links "
  //   << lattice.n_links << " conn_sites " << conn_sites << std::endl ;

  // initialize the spin field
  QfeIsing field(&lattice, beta);
  field.HotStart();
  // printf("initial action: %.12f\n", field.Action());

  std::string data_fname = data_dir + "/"
    + std::to_string(Nx) + "_"
    + std::to_string(Ny) + "_"
    + std::to_string(Nz) + "_"
    + std::to_string(beta) + "_"
    + std::to_string(seed) + ".obs" ;

  std::ofstream data_file ;

  data_file.open(data_fname) ;

  for (int n = 0; n < (n_traj + n_therm); n++) {

    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      int cluster_size = field.WolffUpdate() ;
      // GTF: If cluster_size == 0 it means Wolff tried to update a site with no neighbors.
      // Don't count it as a valid update.
      if (cluster_size < 1) {
        j-- ;
      } else {
        cluster_size_sum += cluster_size ;
      }
    }

    // double action = field.Action() ;

    if (n < n_therm) {
      // std::cout << n << " wolff_sum " << cluster_size_sum << " action " << action << std::endl ;
      continue;
    }

    int m = field.SumSpinGTF();

    std::vector<int> sum_links ;

    field.SumLinkGTF(sum_links, 13) ;

    data_file << n << " " << double(cluster_size_sum) / double(conn_sites) ;

    for (int k : sum_links) {
      data_file << " " << k ;
    }

    data_file << " " << m << std::endl ;

    // GTF START HERE

  }

  data_file.close() ;

  return 0;
}
