// ising_ads_strip.cc

#include <cstdio>
#include <getopt.h>
#include "ads_strip.h"
#include "ising.h"
#include "statistics.h"

// This program measures several bulk quantities of the Ising model on
// an AdS2 "strip" lattice. Measurements occur only on sites in a "lane"
// at the center of the strip. The width of the lane can be changed with
// a command line argument.

// command line arguments:
//   --n_rho sets the total number of lattice sites in the spatial direction
//   --n_t sets the total number of lattice sites in the temporal direction
//   --t_scale sets the ratio of the timelike to spacelike lattice spacing
//   --beta sets the coupling
//   --meas_width determines how wide of a "lane" to measure
//   --n_therm number of thermalization sweeps
//   --n_traj number of sweeps (trajectories) after thermalization
//   --n_skip number of sweeps between measurements
//   --n_wolff number of wolff updates per sweep
//   --n_metropolis number of metropolis updates per sweep

int main(int argc, char* argv[]) {

  // default parameters
  int n_rho = 8;
  int n_t = 8;
  double t_scale = 1.0;
  double beta = 0.4;
  int meas_width = 0;  // default to measuring all sites
  int n_therm = 2000;
  int n_traj = 20000;
  int n_skip = 20;
  int n_wolff = 2;
  int n_metropolis = 5;

  const struct option long_options[] = {
    { "n_rho", required_argument, 0, 'R' },
    { "n_t", required_argument, 0, 'T' },
    { "t_scale", required_argument, 0, 'a' },
    { "beta", required_argument, 0, 'b' },
    { "meas_width", required_argument, 0, 'W' },
    { "n_therm", required_argument, 0, 'h' },
    { "n_traj", required_argument, 0, 't' },
    { "n_skip", required_argument, 0, 's' },
    { "n_wolff", required_argument, 0, 'w' },
    { "n_metropolis", required_argument, 0, 'e' },
    { 0, 0, 0, 0 }
  };

  const char* short_options = "R:T:a:b:W:h:t:s:w:e:";

  while (true) {

    int o = 0;
    int c = getopt_long(argc, argv, short_options, long_options, &o);
    if (c == -1) break;

    switch (c) {
      case 'R': n_rho = atoi(optarg); break;
      case 'T': n_t = atoi(optarg); break;
      case 'a': t_scale = std::stod(optarg); break;
      case 'b': beta = std::stod(optarg); break;
      case 'W': meas_width = atoi(optarg); break;
      case 'h': n_therm = atoi(optarg); break;
      case 't': n_traj = atoi(optarg); break;
      case 's': n_skip = atoi(optarg); break;
      case 'w': n_wolff = atoi(optarg); break;
      case 'e': n_metropolis = atoi(optarg); break;
      default: break;
    }
  }

  printf("n_therm: %d\n", n_therm);
  printf("n_traj: %d\n", n_traj);
  printf("n_skip: %d\n", n_skip);
  printf("n_wolff: %d\n", n_wolff);
  printf("n_metropolis: %d\n", n_metropolis);

  QfeLatticeAdSStrip lattice(n_rho, n_t, t_scale);
  printf("n_rho: %d\n", n_rho);
  printf("n_t: %d\n", n_t);
  printf("t_scale: %.12f\n", lattice.t_scale);
  printf("meas_width: %d\n", meas_width);
  printf("vol: %.12f\n", lattice.vol);

  QfeIsing field(&lattice, beta);
  field.HotStart();
  printf("beta: %.4f\n", field.beta);
  printf("initial action: %.12f\n", field.Action());

  // measurements
  QfeMeasReal mag;  // average spin (magnetization)
  QfeMeasReal mag_2;  // magnetization^2
  QfeMeasReal mag_4;  // magnetization^4
  QfeMeasReal energy;  // energy
  QfeMeasReal energy_2;  // energy^2
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;
  QfeMeasReal accept_overrelax;

  // determine which sites to measure (if zero, measure all sites)
  if (meas_width == 0) meas_width = n_rho;
  int x_start = (n_rho - meas_width) / 2;
  int x_end = n_rho - x_start - 1;

  // calculate the total volume of the measured sites
  double meas_vol = 0.0;
  for (int s = 0; s < lattice.n_sites; s++) {
    int x = s % n_rho;
    if (x < x_start || x > x_end) continue;
    meas_vol += lattice.sites[s].wt;
  }
  printf("meas_vol: %.12f\n", meas_vol);

  for (int n = 0; n < (n_traj + n_therm); n++) {

    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      cluster_size_sum += field.WolffUpdate();
    }
    double metropolis_sum = 0.0;
    for (int j = 0; j < n_metropolis; j++) {
      metropolis_sum += field.Metropolis();
    }
    cluster_size.Measure(double(cluster_size_sum) / double(lattice.n_sites));
    accept_metropolis.Measure(metropolis_sum);

    if (n % n_skip || n < n_therm) continue;

    // spin measurements
    double spin_sum = 0.0;
    for (int s = 0; s < lattice.n_sites; s++) {
      int x = s % n_rho;
      if (x < x_start || x > x_end) continue;
      spin_sum += field.spin[s] * lattice.sites[s].wt;
    }
    double m = spin_sum / meas_vol;
    double m_sq = m * m;
    mag.Measure(fabs(m));
    mag_2.Measure(m_sq);
    mag_4.Measure(m_sq * m_sq);

    // energy measurements
    double energy_sum = 0.0;
    for (int l = 0; l < lattice.n_links; l++) {
      QfeLink* link = &lattice.links[l];
      int s1 = link->sites[0];
      int s2 = link->sites[1];
      double mult = 1.0;
      int x1 = s1 % n_rho;
      int x2 = s2 % n_rho;
      if (x1 < x_start || x1 > x_end) mult -= 0.5;
      if (x2 < x_start || x2 > x_end) mult -= 0.5;
      energy_sum -= field.spin[s1] * field.spin[s2] * link->wt * mult;
    }
    double e = energy_sum / meas_vol;
    energy.Measure(e);
    energy_2.Measure(e * e);

    // print statistics
    printf("%06d %.12f %.4f %.4f\n", \
        n, field.Action(), \
        accept_metropolis.last, \
        cluster_size.last);
  }

  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());

  // print measurements (note: all measurements are per site)
  double m_mean = mag.Mean();
  double m_err = mag.Error();
  double m2_mean = mag_2.Mean();
  double m2_err = mag_2.Error();
  double m4_mean = mag_4.Mean();
  double m4_err = mag_4.Error();

  printf("mag: %.12e %.12e %.4f %.4f\n", \
      m_mean, m_err, mag.AutocorrFront(), mag.AutocorrBack());
  printf("mag^2: %.12e %.12e %.4f %.4f\n", \
      m2_mean, m2_err, mag_2.AutocorrFront(), mag_2.AutocorrBack());
  printf("mag^4: %.12e %.12e %.4f %.4f\n", \
      m4_mean, m4_err, mag_4.AutocorrFront(), mag_4.AutocorrBack());

  double e_mean = energy.Mean();
  double e_err = energy.Error();
  double e2_mean = energy_2.Mean();
  double e2_err = energy_2.Error();

  printf("energy: %.12e %.12e %.4f %.4f\n", \
      e_mean, e_err, energy.AutocorrFront(), energy.AutocorrBack());
  printf("energy^2: %.12e %.12e %.4f %.4f\n", \
      e2_mean, e2_err, energy_2.AutocorrFront(), energy_2.AutocorrBack());

  double U4_mean = 1.5 * (1.0 - m4_mean / (3.0 * m2_mean * m2_mean));
  double U4_err = 0.5 * U4_mean * sqrt(pow(m4_err / m4_mean, 2.0) \
      + pow(2.0 * m2_err / m2_mean, 2.0));
  printf("U4: %.12e %.12e\n", U4_mean, U4_err);

  double m_susc_mean = (m2_mean - m_mean * m_mean);
  double m_susc_err = sqrt(pow(m2_err, 2.0) \
      + pow(2.0 * m_mean * m_err, 2.0));
  printf("m_susc: %.12e %.12e\n", m_susc_mean, m_susc_err);

  double e_susc_mean = (e2_mean - e_mean * e_mean);
  double e_susc_err = sqrt(pow(e2_err, 2.0) \
      + pow(2.0 * e_mean * e_err, 2.0));
  printf("energy_susc: %.12e %.12e\n", e_susc_mean, e_susc_err);

  return 0;
}
