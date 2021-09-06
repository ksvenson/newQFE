// ising.h

#pragma once

#include <cmath>
#include <stack>
#include <cstdio>
#include <vector>
#include "rng.h"

using namespace  std;

struct param {
  int L ;
  int latVol ;
  int Niter ;
  double beta ;
  double axis[3][2];  // a,b,c vectors
  double abc[3];
  double Kfem[3];   // FEM weight on positive links
  int *nn[6];      // Set up nn table nn[site][mu]
  double A;
  double R;
  int *cluster ;
  int *stack ;
  int NumClusters ;
};

void  PrintLatticeInt(int *lat, param p){
  printf("\n x =  :  ");
  for(int x = 0; x < p.L ; x++) printf("  %d  ", x);
  printf("\n");  
  for (int y = 0; y < p.L; y++) {
    printf("y = %d : ", y);
    for (int x = 0; x < p.L ; x++) {
      printf("   %d ",lat[ x + p.L * y]);
    }
    printf("\n");
  }
  printf("\n");
   
}

QfeRng rng(137);

double Area( param p)
{
  return (1.0/4.0)*sqrt( (p.abc[0] + p.abc[1] + p.abc[2]) * (- p.abc[0] + p.abc[1] + p.abc[2]) *
			 (p.abc[0] - p.abc[1] + p.abc[2]) * (  p.abc[0] + p.abc[1] - p.abc[2]) );
}

double Rcircum(param p)
{
  printf("Area =  %f \n", p.A);
  return (p.abc[0] * p.abc[1] * p.abc[2])/( 4.0 * p.A);
}

void HotStart(int *spin, param p) {
  for (int site = 0; site < p.latVol; site++) {
    spin[site] = rng.RandInt(0, 1) * 2 - 1;
  }
}

void ColdStart(int *spin, param p) {
  for (int site = 0; site < p.latVol; site++) {
    spin[site] = 1;
}
}
  
 /*********

Efficient Indexing 

int xp = (x+1)%p.L; 
int yp = (y+1)%p.L;
int  site = x + p.L * y;
int nn = spin[xp + p.L * y] + spin[x + p.L * yp] +  spin[xp + p.L * yp];
action -= p.beta  * spin[site] *  nn;

Can this be done with

inline int nn(int x, int y, int mu)
{
  // mu = 0,1,2,3,4,5 is xp, yp, xyp, -  xp, -  yp,- xyp  
  return  x + 
  }

*********/

int x_mu(int site, int mu, param p)
{
  int nn_site = -1;
  int x = site%p.L;
  int y = site/p.L;
  switch(mu) {
  case 0 :
    nn_site = (x+1)%p.L  + p.L * y;
    break;
  case 1 : 
    nn_site =  x + p.L * ((y + 1)%p.L);
    break;
  case 2 :
    nn_site =  (x+1)%p.L + p.L * ((y + 1 )%p.L);
    break;
  case 3 :
    nn_site = (x-1 + p.L)%p.L + p.L * y;
    break;
  case 4 :
    nn_site =  x + p.L * ((y- 1 + p.L)%p.L);
    break;
  case 5 :
    nn_site =   (x-1 + p.L)%p.L + p.L * ((y- 1 +p.L)%p.L);
    break;
  default :
    printf("Invalid neighbor index\n" );
  }
  return nn_site;
}


double IsingAction(int *spin, param p) {
  double action = 0.0;
  double nn = 0.0;
  // sum over links
  for (int y = 0; y < p.L; y++) {
    for (int x = 0; x < p.L ; x++) {
      nn = p.Kfem[0]*spin[(x+1)%p.L  + p.L * y] +  p.Kfem[1]*spin[x + p.L * ((y+1)%p.L)]
	+   p.Kfem[2]*spin[(x+1)%p.L + p.L * ((y+1)%p.L)];
      action = - p.beta  *  spin[ x + p.L * y] *  nn;  
    }
  } 
  return action/double(p.latVol);
}

/* Tricky Geometry: Need to sum rvec = delta_x avec + delta_y cvec  and
   clip rvec to be inside a hexagon and the cal sqrt(rvec.rvec).
   
   Rule x \in [- L/2,L/2]  &  y \in [-L/2,L/2]  & x y > 0
   
*/

double Mod(int xp, int x, int L)
{
  double DeltaX = 0.0;
  
  if(abs(xp - x) <= L/2)
    {
      DeltaX = xp - x;
    }
  else
    {
      DeltaX = (xp - x) > 0 ?   (xp - x) - L:   (xp - x) + L;
    }
  return DeltaX;
}
// Need to retrun the vector as well

double distance(int site1, int site2, param p)
{
  double rvec[2];
  double delta_x = Mod(site1%p.L,site2%p.L,p.L);
  double delta_y = Mod(site1/p.L, site2/p.L, p.L); 
  rvec[0] =  delta_x * p.axis[0][0] +  delta_y * p.axis[1][0];
  rvec[1] =  delta_x * p.axis[0][1] +  delta_y * p.axis[1][1];
  return sqrt( rvec[0]*rvec[0] + rvec[1]*rvec[1]);
}  


double MeanSpin(int *spin, param p) {
  double m = 0.0;
  for (int s = 0; s < p.latVol; s++) {
    m += spin[s];
  }
  return m;
}

  // metropolis update algorithm
// ref: N. Metropolis, et al., J. Chem. Phys. 21, 1087 (1953).

double Metropolis(int *spin, param p) {
  int accept = 0;
  double delta_S = 0.0;
  double h = 0.0;
  
  // sum over links
    for (int y = 0; y < p.L; y++) {
      for (int x = 0; x < p.L ; x++) {
      int spinxy = spin[ x + p.L * y];
      h = 0.0;
      // sum over links connected to this site
      h +=   p.Kfem[0]*( spin[(x+1)%p.L  + p.L * y]   +  spin[(x-1 + p.L)%p.L  + p.L * y]);
      h  +=  p.Kfem[1]*(spin[x + p.L * ((y + 1)%p.L)]   +  spin[x + p.L * ((y- 1 + p.L)%p.L)]);
      h  +=  p.Kfem[2] * (spin[(x+1)%p.L + p.L * ((y + 1 )%p.L)] + spin[(x-1 + p.L)%p.L + p.L * ((y- 1 +p.L)%p.L)]);
      
      delta_S = 2.0 * p.beta* spinxy * h;
   
      //  PrintLatticeInt(spin,  p);
      // metropolis algorithm: Minimize delta_S =  S_final -  S_initial
      //  if (delta_S <= 0.0 ||rng.RandReal() < exp(-delta_S)) {
       if (rng.RandReal() < exp(-delta_S)) {
	spin[ x + p.L * y] = -spinxy;
	//	printf("Flip x =  %d, y = %d , delta_S =  %0.8f \n", x,y,delta_S);
	//	 PrintLatticeInt(spin,  p);
	accept++;
      }
    }
  }
  return double(accept) / double(p.latVol);
}

// wolff cluster update algorithm
// ref: U. Wolff, Phys. Rev. Lett. 62, 361 (1989).

int WolffUpdate(int *spin, param p){
  int site_mu = -1;
  int  cluster_size = 0;
  bool is_clustered[p.latVol];
  for(int i = 0 ; i< p.latVol;i++)  is_clustered[i] = false;
  
  std::stack<int> stack; // create the stack called stack
  int s = rng.RandInt(0, p.latVol - 1);  // choose a random site and add to the stack/cluster
  // wolff_cluster.push_back(s);
  is_clustered[s] = true;
  cluster_size++;  // add to cluster size
  stack.push(s);  // printf("Push  site = %d \n", s);

  while (stack.size() != 0){
    s = stack.top();  // take out of stack -- never again
    stack.pop(); 
    int value = spin[s]; // printf("Pop  site = %d \n",s);
    spin[s] = - value; //flip it
   
    
    // try to add neighbors
    for (int mu = 0; mu < 6; mu++) {
      site_mu= x_mu(s , mu,  p);

      // skip if the site is already clustered
      if (is_clustered[site_mu]) continue;
      
      // skip if sign bits don't match
      if (value != spin[site_mu] ) continue;
      
      double prob = 1.0 - exp(-2.0 * p.Kfem[mu % 3]* p.beta );
      if (rng.RandReal() < prob) {
	//	wolff_cluster.push_back(s);  // add the site to the cluster
	 is_clustered[site_mu] = true;
	  cluster_size++; 
	  stack.push(site_mu); // printf("Push  site = %d \n", site_mu);
      }
    }
  }
  
  // Only need this or something like it for correlator: What about Fourrier space.
  // return  wolff_cluster.size(); 

  return  cluster_size;
  
}
/* This routine keep a list of the sites in the cluster.
can be combined with Wollf Update with flag.
*/
  int WolffUCorrelator(int *spin,int  *cluterList, param p){
    int site_mu = -1;
  int  cluster_size = 0;
  bool is_clustered[p.latVol];
  int  index_cluster[p.latVol];
  for(int i = 0 ; i< p.latVol;i++)  is_clustered[i] = false;
  
  std::stack<int> stack; // create the stack called stack
  int s = rng.RandInt(0, p.latVol - 1);  // choose a random site and add to the stack/cluster
  // wolff_cluster.push_back(s);
  is_clustered[s] = true;
  index_cluster[cluster_size] = s;
  cluster_size++;  // add to cluster size
  stack.push(s);  // printf("Push  site = %d \n", s);

  while (stack.size() != 0){
    s = stack.top();  // take out of stack -- never again
    stack.pop(); 
    int value = spin[s]; // printf("Pop  site = %d \n",s);
    spin[s] = - value; //flip it
   
    
    // try to add neighbors
    for (int mu = 0; mu < 6; mu++) {
      site_mu= x_mu(s , mu,  p);

      // skip if the site is already clustered
      if (is_clustered[site_mu]) continue;
      
      // skip if sign bits don't match
      if (value != spin[site_mu] ) continue;
      
      double prob = 1.0 - exp(-2.0 * p.Kfem[mu % 3]* p.beta );
      if (rng.RandReal() < prob) {
	//	wolff_cluster.push_back(s);  // add the site to the cluster
	 is_clustered[site_mu] = true;
	 index_cluster[cluster_size] = site_mu;
	  cluster_size++;
	  index_cluster[cluster_size] = site_mu;
	  stack.push(site_mu); // printf("Push  site = %d \n", site_mu);
      }
    }
  }
  
  // Only need this or something like it for correlator: What about Fourrier space.
  // return  wolff_cluster.size(); 

  return  cluster_size;
  
}


      
