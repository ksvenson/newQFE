/* 
(1) Goal Maximally Simple FEM Ising on a flat Triangular lattice
with arbitray length {a,b,c} = axes[3]

(2) Model of Efficient Data Parallel Code.
Do Metropolis, Wolff and Swendens Wang by HK algorithm.

(3) NO C++ but take from Evan element when and if useful
from newFEM/ising_s2_crit. and later for our old FEM routines. 

Step 1: Implement Triangle/Square lattice and scan Binder Cumlant for
critical point

Sept 2: put in Skew and find critcal point.

Step 3: Look at correlator for spherial Symmetry. (How ? Fourier modes?)


*/

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
//#include <iostream>
#include "ising.h"
#include "statistics.h"
using namespace  std;

int main()
{  
  struct param p;

  // BEGIN SET PARAMETERS 
  p.L =  64;
  p.latVol = p.L * p.L;
  // p.beta = 0.27465307; // log(3)/4 
  p.beta =  0.951426150896; // FEM scaled 2.0 *sqrt(3) * log(3)/4.0 for triangular case.
  //  p.beta = 0.0;
  int spin[p.latVol];

  ColdStart(spin,p);
  //PrintLatticeInt(spin,p);
 
  double  vec[][2] = {
    {1.0,0.0},
    {-0.5,sqrt(3.)/2.0},
    {- 0.5,- sqrt(3.)/2.0}
  };

  for(int i=0; i<3; i++) {
   for(int j=0;j<2;j++) 
     p.axis[i][j] = vec[i][j];
   };
  
  for(int i = 0; i< 3; i++){
    p.abc[i] = sqrt( p.axis[i][0] * p.axis[i][0]+ p.axis[i][1] * p.axis[i][1] );
    printf("abc = %.12f \n" , p.abc[i]);
  }
  p.A = Area(p);  printf("Area =  %.20f \n", p.A);
  p.R = Rcircum(p);  printf("Rcircum =  %.12f \n", p.R);

  for(int mu = 0; mu <3; mu++) {
    //  p.Kfem[mu] = 0.5 * sqrt((4.0 * p.R * p.R)/(p.abc[mu] * p.abc[mu]) -1.0 );
    p.Kfem[mu] = 0.5 * sqrt((4.0 * p.R * p.R)/(p.abc[mu] * p.abc[mu]) -1.0 );
      printf("p.Kfem = %.12f ",  p.Kfem[mu]);
  }

 // END SET PARAMETERS parm p;

 
#if 0   // TEST  Distance Tests & Indexing
  for(int site1 = 0; site1 < p.latVol; site1++){
    printf("i = %d \n",site1);
    for(int site2 = 0; site2 < p.latVol; site2++)
      {
  double rvec[2];
  double delta_x = Mod(site1%p.L,site2%p.L,p.L);
  double delta_y = Mod(site1/p.L, site2/p.L, p.L); 
  rvec[0] =  delta_x * p.axis[0][0] +  delta_y * p.axis[1][0];
  rvec[1] =  delta_x * p.axis[0][1] +  delta_y * p.axis[1][1];
  printf("j :  %d, r = ( %0.4f,%0.4f ) \n", site2, rvec[0],  rvec[1]);
    }
  }

  for(int site1 = 0; site1 < p.latVol; site1++){
    printf("i = %d",site1);
    for(int site2 = 0; site2 < p.latVol; site2++)
      {
	printf(" %.4f", distance(site1, site2,  p) );
      }
    printf(" \n ");
  }

  printf(" \n ");
  for(int site = 0; site < p.latVol; site++){
    for(int mu = 0; mu < 6 ; mu++) {
      printf(" mu = %d,  x = %d   y = %d site_x = %d  x_mu = %d \n",mu,  site%p.L, site/p.L, site,  x_mu(site,  mu, p));
    }
  }
  #endif
  
    HotStart(spin, p);
// ColdStart(spin,p);

//PrintLatticeInt(spin,p);

 // measurements
  std::vector<double> mag;
  double action = 0.0;
  double action_sum = 0.0;
  double accept = 0.0;
  double accept_sum = 0.0;
  int cluster = 0;
  int cluster_sum = 0;
  int count = 0;
  
 
  int n_therm = 1000;
  int n_traj = 20000;
  int n_skip = 2;
  int n_wolff = 3;
  int n_metropolis = 5;

  
  for (int n = 0; n < (n_traj + n_therm); n++) {
    for (int j = 0; j < n_wolff; j++) {
      cluster = WolffUpdate(spin, p);
    }
    for (int j = 0; j < n_metropolis; j++) {
      accept = Metropolis(spin, p);
    }
 
    if (n % n_skip || n < n_therm) continue;
     action =  IsingAction(spin, p);
     accept_sum += accept;
     cluster_sum += cluster;
     action_sum += action;
      count++;
      if(n%500 == 0) printf(" accept = %0.4f  size = %d action = %.12f \n", accept, cluster, action);
      
  }
  

  printf("Averages: accept = %0.4f  size = %.12f action = %.12f \n", accept_sum/double(count), cluster_sum/double(count), action_sum/double(count));

  
  return 0;
}


