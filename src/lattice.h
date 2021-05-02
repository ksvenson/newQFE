// lattice.h

#pragma once

#include <vector>
#include <random>
#include "rng.h"

struct QfeSite {
  int id;  // site id
  double wt;  // site weight
  std::vector<int> links;  // ids of links connected to this site
  std::vector<int> neighbors;  // ids of sites connected to this site
};

struct QfeLink {
  int id;  // link id
  double wt;  // link weight

  // id of sites that this link goes between
  int sites[2];
};

class QfeLattice {

 public:
  QfeLattice();
  void InitTriangle(int N, double skew = 0.0);
  QfeLink AddLink(int a, int b, double wt);
  double Action();
  void HotStart();
  void ColdStart();

  inline QfeSite GetSite(int s) { return sites[s]; }
  inline QfeLink GetLink(int l) { return links[l]; }
  inline double GetPhi(int s) { return phi[s]; }
  inline void SetPhi(int s, double value) { phi[s] = value; }
  inline int n_sites() { return sites.size(); }
  inline int n_links() { return links.size(); }
  inline double GetMag() { return mag; }
  inline double GetMusq() { return musq; }
  inline double GetLambda() { return lambda; }
  inline QfeRng* GetRng() { return &rng; }

 private:
  // general qfe lattice variables
  std::vector<QfeSite> sites;
  std::vector<QfeLink> links;
  QfeRng rng;

  // phi^4 specific variables
  std::vector<double> phi;  // phi field
  double mag;  // magnetization per site (sum_x phi_x / n_sites)
  double lambda;  // bare coupling
  double musq;  // bare mass squared
};
