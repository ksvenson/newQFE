// statistics.h

#pragma once

#include <vector>

double Mean(std::vector<double>& a);
double LogMean(std::vector<double>& a);
double U4(std::vector<double>& m2, std::vector<double>& m4);
double Susceptibility(std::vector<double>& m2, std::vector<double>& m_abs);
double JackknifeMean(std::vector<double>& a);
double JackknifeLogMean(std::vector<double>& a);
double JackknifeU4(std::vector<double>& m2, std::vector<double>& m4);
double JackknifeSusceptibility(std::vector<double>& m2, std::vector<double>& m_abs);
double AutocorrGamma(std::vector<double>& a, int n);
double AutocorrTime(std::vector<double>& a);
