#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <cmath>
#include <complex>
#include <vector>
#include <cstring>

using namespace std;

//Library wide precision.
typedef long double Float;

#include "util.h"
#include "hyp_util.h"
#include "graph.h"

//If true, prints a bunch of numbers!
bool flags = false;

//Utilities
void copyLattice(vector<Vertex> &Lattice, bool new_to_old, param p);
void zeroField(Float *field, param p);
Float measH(vector<Vertex> Lattice, Float *mom, param p);
void gaussReal(Float *mom, param p);

//HMC routines
int hmc(vector<Vertex> &Lattice, param p, int iter);
void forceU(Float *fU, vector<Vertex> &Lattice, param p);
void update_mom(Float *mom, Float *fU, param p, double dt);
void update_phi(vector<Vertex> &Lattice, Float *mom, param p, double dt);
void trajectory(vector<Vertex> &Lattice, Float *mom, param p);

//Heatbath
void heatbath(vector<Vertex> &Lattice, param p, int iter);

//Propagator
Float compute_G(vector<Vertex> &Lattice, int n, int N);

Float dHAve = 0.0;
int accepted_metropolis = 0;
 
// Begin Main Program
//==============================================================
int main(int argc, char **argv) {
  
  param p;
  //Process Command line arguments
  for (int i=1; i<argc; i++){
    if(p.init(argc, argv, &i) == 0){
      continue;
    }
    printf("ERROR: Invalid option: %s\n", argv[i-1]);
    p.usage(argv);
    exit(0);
  }

  int k=0;
  if(argc > 1) p.init(argc, argv, &k);
  
  //Pseudo RNG seed
  srand48(p.seed);
  
  p.S1 = endNode(p.Levels,p) + 1 - (endNode(p.Levels-1,p) + 1);
  p.surfaceVol = p.S1*p.Lt;
  p.AdSVol = endNode(p.Levels,p) + 1;
  p.latVol = p.AdSVol;
  
  //Print paramters
  p.print();
  
  //Object to hold graph data
  vector<Vertex> Lattice(p.latVol);
  
  //-1 in Lattice indicates that node n has no connections.
  //This is used during construction to indicate if the node is yet
  //to be populated. During truncation, nodes to be removed are
  //assigned a position of -1, and all connections to that node are
  //removed.
  for(int n = 0; n < p.latVol; n++)
    for(int mu = 0; mu < p.q+2; mu++) 
      Lattice[n].nn[mu] = -1;
  
  //Construct neighbour table.
  buildGraph(Lattice, p);
  //Get the z-coords
  getComplexPositions(Lattice, p);

  //Begin (H)MC routines
  //---------------------------------------------------------------------------

  //Fake out
  //If we want to compute the lattice action, we need a momentum field.  
  Float mom[p.AdSVol];
  zeroField(mom, p);

  int accept = 0;
  int accepted = 0;
  int count = 0;  //proxy for number of thermalizations
  
  //Initialize Lattice
  for(int i = 0; i<p.latVol; i++) {
    Lattice[i].phi = 2.0*drand48() - 1.0;
    if(drand48() < 0.5) Lattice[i].ising = -1;
    else Lattice[i].ising = 1;
  }
  //copyLattice(Lattice, true, p);
  //Float test[1][2][3];
  cout<<"into the woods"<<endl;
  
  //Initialize propagator G
  //Need of form G[count][i][j]
  Float***G = new Float**[p.n_meas];
  Float**Gavg = new Float*[p.latVol];

  for(int i=0; i<p.latVol; i++)
    {
      Gavg[i] = new Float[p.latVol];
      for(int j=0; j<p.latVol; j++)
	{
	  Gavg[i][j] = 0.0;
	}
    }
  
  for(int n=0; n<p.n_meas; n++)
    {
      G[n] = new Float*[p.latVol];
      //cout<<"here"<<endl;
      //G[i][j] = new Float*[p.latVol];
      for(int i=0; i<p.latVol; i++)
	{
	  G[n][i] = new Float[p.latVol];
	  //G[n][i] = new Float[p.latVol];
	  for(int j=0; j<p.latVol; j++)
	    {
	      G[n][i][j] = 0.0;
	    }
	}
    }

  cout<<"out of the woods"<<endl;
  
  //for(int n=0; n<p.nmeas; n++)
  //{
  //  Gcount[n] = G[i][j];
  //}
  
  //Loop over warmup iterations
  for(int iter=0; iter<p.n_therm; iter++) {
    accept = hmc(Lattice, p, iter);    
    heatbath(Lattice, p, iter);
  }
  //Loop over measurement iterations
  for(int iter=p.n_therm; iter<p.n_therm + p.n_meas-1; iter++) {
    //cout<<"count is at beginning: "<<count<<endl;
    count++;
    //cout<<"count is now: "<<count<<endl;
    //double Hold = measH(Lattice, mom, p);
    heatbath(Lattice, p, iter);
    //double H = measH(Lattice, mom, p);
    //accepted += hmc(Lattice, p, iter);
    //cout << iter << " " << setprecision(16) << (1.0*accepted_metropolis)/((iter+1)*p.AdSVol) << " " << H << " " << (H-Hold) << endl; 

    //Compute boundary-boundary propagator
    for(int i=0; i<p.latVol; i++)
      {
	for(int j=0; j<p.latVol; j++)
	  {
	    G[count][i][j] = Lattice[i].phi*Lattice[j].phi;
	  }
      }

    

      
    //G[count][n] = compute_G(Lattice, n, p.latVol);

      //cout<<"G["<<count<<"]["<<n<<"] is: "<<G[count][n]<<endl;
  }
    //cout<<"count is "<<count<<endl;

  //Average over MC runs
 
  for(int i=0; i<p.latVol; i++)
    {
      for(int j=0; j<p.latVol; j++)
	{
	  for(int n=0; n<p.n_meas; n++)
	    {
	      Gavg[i][j] += G[n][i][j]/p.n_meas;
	    }
	}
    }
  
  
  cout<<"test"<<endl;
  /*  
  Float* Gtemp = new Float[p.n_meas];
  Float* Gavg = new Float[p.n_meas];
  for(int i=0; i<p.n_meas; i++) {
    Gtemp[i] = 0.0;
    Gavg[i] = 0.0;
  }
  
  for(int i=0; i<p.latVol; i++) {
    for(int j=0; j<count; j++) {
      Gtemp[i] += G[j][i];
    }
    Gavg[i] = Gtemp[i]/(1.0*count);
  }
  */
  cout<<"Print to file?"<<p.src_pos<<endl;
  
  //Print to file
  ofstream myfile;
  myfile.open("therm_bound_bound.dat");
  for(int i=endNode(p.Levels-1,p)+1; i<endNode(p.Levels,p)+1; i++) {
    complex<long double> ratio = Lattice[i].z/Lattice[p.src_pos].z;
    long double theta = atan2(ratio.imag(), ratio.real());
    myfile<<d12(Lattice[p.src_pos].z, Lattice[i].z)<<" "
	  <<theta<<" "
      //<<Gavg[i]<<" "
	  <<abs(Lattice[i].z)<<" "
	  <<Lattice[i].z.real()<<" "
	  <<Lattice[i].z.imag()<<" "
	  <<Lattice[p.src_pos].z.real()<<" "
	  <<Lattice[p.src_pos].z.imag()<<"\n";
  }
  myfile.close();
  //1 geodesic distance
  //2 theta
  //3 MC lattice propagator all-to-all
  //4 |z|
  //5 Re[z]
  //6 Im[z]


  ofstream myfile1;
  myfile1.open("therm_G.dat");
  for(int i=0; i<p.latVol; i++)
    {
      for(int j=0; j<p.latVol; j++)
	{
	  myfile1 << Gavg[i][j] << " ";
	}
      myfile1 << "\n"; 
    }
  myfile1.close();


  ofstream myfile2;
  myfile2.open("zcoords.dat");
  for(int i=0; i<p.latVol; i++)
    {
      complex<long double> ratio = Lattice[i]
      long double theta = atan2(ratio.imag(), ratio.real());
      myfile2 <<Lattice[i].z.real()<<" "<<Lattice[i].z.imag()<<"\n"; 
    }
  myfile2.close();

    
  
  return 0;
}

Float compute_G(vector<Vertex> &Lattice, int n, int N) {
  double phi_temp = 0.0;
  for(int j=0; j<N; j++) phi_temp += Lattice[j].phi*Lattice[n].phi;
  return phi_temp/(1.0*N);
}


void heatbath(vector<Vertex> &Lattice, param p, int iter) {
  
  double phi_new = 0.0;
  double phi_new_sq = 0.0;
  double phi = 0.0;
  double phi_sq = 0.0;
  double lambda = 0.25*p.lambda;
  double msqr   = 0.50*p.msqr;
  
  double deltaH = 0.0;

  //cout<<"beginning heatbath iter "<<iter<<endl;
  
  for (int i=0; i<p.latVol; i++) {
    
    deltaH = 0.0;
    
    phi = Lattice[i].phi;
    phi_new = phi + p.delta_phi * (2.0*drand48() - 1.0);
    phi_new_sq = phi_new*phi_new;
    phi_sq = phi*phi;
    
    //PE
    deltaH += lambda*(phi_new_sq*phi_new_sq - phi_sq*phi_sq);
    deltaH += msqr  *(phi_new_sq            - phi_sq);
    
    //KE
    for(int q=0; q<p.q; q++) {
      if(Lattice[i].nn[q] != -1) {
	deltaH += 0.5 * (phi_new_sq - phi_sq + 2*Lattice[Lattice[i].nn[q]].phi*(phi - phi_new));
      }
    }

    if(deltaH < 0.0) {
      int temp = Lattice[i].ising;
      //cout << "Accepted" << endl;
      accepted_metropolis++;
      Lattice[i].phi = phi_new;
      Lattice[i].ising = -temp;
    }
    else if ( drand48() < exp(-deltaH)) {
      //cout<< "Acepted" << endl;
      accepted_metropolis++;
      Lattice[i].phi = phi_new;
    }
  }
  //cout<<"end heatbath iter "<<iter<<endl;
}

// Utilities
//----------------------------------------------------------------------------

//Computes the action of the field
Float measH(vector<Vertex> Lattice, Float *mom, param p) {

  Float KE, PE, Hmom;
  Float phi_sq;
  Float phi;
  Float lambda_p = 0.25*p.lambda;
  Float msqr_p   = 0.50*p.msqr;

  for (int i=1; i<p.latVol; i++) {

    //Field action
    //------------
    phi = Lattice[i].phi;
    phi_sq = phi*phi;
    
    //PE terms
    PE += lambda_p * phi_sq*phi_sq;
    PE += msqr_p   * phi_sq;
    
    //Spatial: q=0 and q=fwdLinks+1 are on the same level. We take q=fwdLinks+1 and
    //the fwdLinks as the forward links in a forward difference operator
    for(int q=0; q<p.q; q++) {
      //for(int q=0; q<Lattice[i].fwdLinks+1; q++) {
      
      //Test if the link is connected
      if(Lattice[i].nn[q] != -1) {
	
	//Compute kinetic term
	KE += 0.5*((phi - Lattice[Lattice[i].nn[q]].phi)*
		   (phi - Lattice[Lattice[i].nn[q]].phi));
      }
    }
    
    //Momentum action
    //---------------
    Hmom += 0.5 * mom[i]*mom[i];
  }
  
  if (flags) cout << KE << " " << PE << " " << Hmom << " ";
  return KE + PE + Hmom;
}

//normalized gaussian exp[-phi*phi/2] |  <phi^2> = 1
void gaussReal(Float *field, param p) {  
  double r, theta, sum;
  for(int i=0; i<p.AdSVol; i++) {
    r = sqrt(-2.0*log(drand48()));
    theta = 2*M_PI * drand48();
    field[i] = r*cos(theta);
  }
  
  return;
}

void copyLattice(vector<Vertex> &Lattice, bool new_to_old, param p){
  
  if (new_to_old) {    
    for (int i=0; i<p.AdSVol; i++) Lattice[i].phi_old = Lattice[i].phi;
  } else {
    for (int i=0; i<p.AdSVol; i++) Lattice[i].phi = Lattice[i].phi_old;
  }
}

void zeroField(Float *field, param p){
  
  for (int i=0; i<p.AdSVol; i++) field[i] = 0.0;
}







// HMC routines
//----------------------------------------------------------------------------

int hmc(vector<Vertex> &Lattice, param p, int iter) {

  int accepted = 0;
  Float Hold = 0, H = 0;
  
  Float mom[p.AdSVol];
  gaussReal(mom, p);

  // MD trajectory using Verlet (Leapfrog)
  Hold = measH(Lattice, mom, p); if(flags) cout << endl;
  trajectory(Lattice, mom, p);
  H = measH(Lattice, mom, p); if(flags) cout << endl;
  
  if (iter+1 > p.n_therm) {
    
    // Metropolis accept/reject step.

    if( (H-Hold) != (H-Hold)) {
      cout << "Nope..." << endl;
      exit(0);
    } else {
      cout << "deltaH = " << (H-Hold) << endl;
    }
    
    dHAve += (H-Hold);
    if ( drand48() > exp(-(H-Hold)) ) {
      //Rejected
      //cout << "Rejecting " << exp(-(H-Hold)) << endl;
      copyLattice(Lattice, false, p);
    }
    else {
      //Accepted
      //cout << "Accepting " << exp(-(H-Hold)) << endl;
      accepted = 1;
      copyLattice(Lattice, true, p);
    }
  } else {
    //Auto accept
    copyLattice(Lattice, true, p);    
  } 
  return accepted;
  
}

void trajectory(vector<Vertex> &Lattice, Float *mom, param p) {

  const int n_step = p.n_step;
  const Float dt = p.tau/n_step;

  Float fU[p.AdSVol];
  
  if (flags) cout << 0 << " " << measH(Lattice, mom, p) << endl;
  
  // Implement the Leapfrog method
  //Initial half step:
  //P_{1/2} = P_0 - dtau/2 * fU  
  forceU(fU, Lattice, p);
  update_mom(mom, fU, p, 0.5*dt);
  if (flags) cout << 0.5 << " " << measH(Lattice, mom, p) << endl;
  //step loop
  for(int k=1; k<n_step; k++) {
    
    //U_{k} = U_{k-1} + P_{k-1/2} * dt
    update_phi(Lattice, mom, p, dt);
    
    //P_{k+1/2} = P_{k-1/2} - fU * dt
    forceU(fU, Lattice, p);
    update_mom(mom, fU, p, dt);

    if (flags) cout << k << " " << measH(Lattice, mom, p) << endl;
    
  } //end step loop
  
  //Final half step.
  //U_{n} = U_{n-1} + P_{n-1/2} * dt
  update_phi(Lattice, mom, p, dt);
  forceU(fU, Lattice, p);
  update_mom(mom, fU, p, 0.5*dt);

  if (flags) cout << "final" << " " << measH(Lattice, mom, p) << endl;
}

void forceU(Float *fU, vector<Vertex> &Lattice, param p) {

  zeroField(fU, p);
  const Float msqr  = 0.5*p.msqr;
  const Float lambda = 0.25*p.lambda;
  Float phi_lc = 0.0;
  
  for(int i=1; i<p.AdSVol; i++) {
    
    //if( i < endNode(p.Levels-1, p) +1) {
      
      //A convenience
      phi_lc = Lattice[i].phi;
      //cout << i << " ";
      for (int q=0; q<p.q; q++) {
	//cout << Lattice[i].nn[q] << " ";
	if(Lattice[i].nn[q] != -1) {
	  //cout << Lattice[i].nn[q] << " ";
	  fU[i] -= 1.0*(Lattice[Lattice[i].nn[q]].phi - phi_lc);
	}
	else {
	  fU[i] += 1.0*phi_lc;
	}
      }
      //cout << endl;
      fU[i] += (2.0*msqr + 4.0*lambda*phi_lc)*phi_lc*phi_lc;
    }
  //}  
}

void update_mom(Float *mom, Float *fU, param p, double dt) {

  for(int i=0; i<p.AdSVol; i++) mom[i] -= fU[i] * dt;
  
}

void update_phi(vector<Vertex> &Lattice, Float *mom, param p, double dt) {
  
  for(int i=0; i<p.AdSVol; i++) Lattice[i].phi += mom[i] * dt;
  
}
