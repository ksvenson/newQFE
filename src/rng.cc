// rng.cc

#include "rng.h"

QfeRng::QfeRng(int seed) {
  gen = std::mt19937(seed);
}

double QfeRng::RandReal(double min, double max) {
  std::uniform_real_distribution<double> dist(min, max);
  return dist(gen);
}

double QfeRng::RandNormal(double mean, double stddev) {
  std::normal_distribution<double> dist(mean, stddev);
  return dist(gen);
}

int QfeRng::RandInt(int min, int max) {
  std::uniform_int_distribution<int> dist(min, max);
  return dist(gen);
}
