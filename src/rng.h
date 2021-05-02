// rng.h

#pragma once

#include <random>

class QfeRng {

 public:
   QfeRng(int seed = 12345678);
   double RandReal(double min = 0.0, double max = 1.0);
   double RandNormal(double mean = 0.0, double stddev = 1.0);
   int RandInt(int min, int max);

 private:
   std::mt19937 gen;
};
