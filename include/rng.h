// rng.h

#pragma once

#include <random>

using std::mt19937;
using std::uniform_real_distribution;
using std::normal_distribution;
using std::uniform_int_distribution;

class QfeRng {

public:
  QfeRng(int seed = 12345678);
  double RandReal(double min = 0.0, double max = 1.0);
  double RandNormal(double mean = 0.0, double stddev = 1.0);
  int RandInt(int min, int max);

  mt19937 gen;
};

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
