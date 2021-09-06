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
#include <iostream>
//#include "param.h"
#include "ising.h"
#include "statistics.h"
using namespace  std;

int main()
{  
  

  // BEGIN SET PARAMETERS   MOVE TO param.h file

  param p;

  p.L =  64;
  p.latVol = p.L * p.L;
  // p.beta = 0.27465307; // log(3)/4 
  p.beta =  0.951426150896; // FEM scaled 2.0 *sqrt(3) * log(3)/4.0
  int spin[p.latVol];
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


  
#if 0   // TEST  Distance Tests
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
#endif


  // measurements
  std::vector<double> mag;
  std::vector<double> action;
  QfeMeasReal cluster_size;
  QfeMeasReal accept_metropolis;

  int n_therm = 10000;
  int n_traj = 200000;
  // int n_skip = 2;
  // int n_wolff = 3;
  int n_skip = 21;
  int n_wolff = 1;
  //int n_metropolis = 5;
  int n_metropolis = 1;
  for (int n = 0; n < (n_traj + n_therm); n++) {

    int cluster_size_sum = 0;
    for (int j = 0; j < n_wolff; j++) {
      cluster_size_sum =  WolffUpdate(spin,p);
      	if(n%1000 ==0) printf("Cluster size = %d \n",cluster_size_sum);
    }
    double metropolis_sum = 0.0;
    for (int j = 0; j < n_metropolis; j++) {
     metropolis_sum = Metropolis(spin,p);
      	if(n%1000 ==0) printf("Metropois Acceptance  %.12f \n", metropolis_sum);      
    }
    cluster_size.Measure(double(cluster_size_sum) / double(p.L * p.L));
    accept_metropolis.Measure(metropolis_sum);

    if (n % n_skip || n < n_therm) continue;

    action.push_back(IsingAction(spin,p));
    mag.push_back(MeanSpin(spin,p));
    /*     printf("%06d %.12f %+.12f %.4f %.4f\n",	\
        n, action.back(), mag.back(), \
        accept_metropolis.last, \
        cluster_size.last);
    */
  }

  std::vector<double> mag_abs(mag.size());
  std::vector<double> mag2(mag.size());
  std::vector<double> mag4(mag.size());
  for (int i = 0; i < mag.size(); i++) {
    double m = mag[i];
    double m2 = m * m;
    mag_abs[i] = fabs(m);
    mag2[i] = m2;
    mag4[i] = m2 * m2;
  }

  #if 0
  printf("accept_metropolis: %.4f\n", accept_metropolis.Mean());
  printf("cluster_size/V: %.4f\n", cluster_size.Mean());
  printf("action: %.12e (%.12e), %.4f\n",				\
      Mean(action), JackknifeMean(action), AutocorrTime(action));
  printf("m: %.12e (%.12e), %.4f\n", \
      Mean(mag), JackknifeMean(mag), AutocorrTime(mag));
  printf("m^2: %.12e (%.12e), %.4f\n", \
      Mean(mag2), JackknifeMean(mag2), AutocorrTime(mag2));
  printf("m^4: %.12e (%.12e), %.4f\n", \
      Mean(mag4), JackknifeMean(mag4), AutocorrTime(mag4));
  printf("U4: %.12e (%.12e)\n", U4(mag2, mag4), JackknifeU4(mag2, mag4));
  printf("susceptibility: %.12e (%.12e)\n", Susceptibility(mag2, mag_abs), \
      JackknifeSusceptibility(mag2, mag_abs));
#endif
  return 0;
}

 


