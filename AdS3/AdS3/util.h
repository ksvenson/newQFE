#ifndef UTIL_H
#define UTIL_H
#include <complex>
#include <cstring>
#include <math.h>

//#include "gsl_fit.h"

using namespace std;

#define I complex<Float>(0.0,1.0)

class Param{

 public:

  int q = 7;
  
  bool bc = true;       //if true, use Dirichlet. If false, use Neumann
  bool Vcentre = true;  //if true, place vertex at centre. If false, use circumcentre.
  bool verbosity = false;  //if true, print all data. If false, print summary.
  int MaxIter = 100000;
  int n_shift = 100;
  Float tol = pow(10,-6);
  Float msqr = 1.0;
  Float lambda = 0.0;
  Float C_msqr = 1.0;
  Float N_latt = 1.0;
  int Levels = 2;
  int src_pos = -1;
  Float DiskScale = 1.0;
  char fname[256];

  int at = 0.1;
  //int omega = 0;
  int Lt = 56;
  int S1 = 32;
  int SurfaceVol = 0;
  int AdSVol = 0;
  int latVol = 0;
  //double lambda = 1.0;
  double musqr = 1.0;
  int *cluster ;    // Swendsen Wang Data Struture
  int *stack ;     // Wolf Data Struture
  int NumClusters ;

  //HMC 
  int n_metro_cool = 0;
  int n_therm = 1000;
  int n_meas = 1000;
  int n_write = 100;
  int n_skip = 100;
  int n_cluster = 8;
  int n_jkblock = 10;
  int n_step = 10;
  double tau = 1.0;
  double dt = 0.1;
  double delta_phi = 1.5;

  int t=2;
  
  void print()
  {
    cout<<"**********************************************************"<<endl;
    cout<<"Parameter status:"<<endl;
    cout<<"Triangulation = "<<q<<endl;
    cout<<"B.C. = "<< (bc ? ("Dirichlet") : ("Neumann") ) << endl;
    cout<<"Centre = "<< (Vcentre ? ("Vertex") : ("Circum") ) << endl;
    cout<<"MaxIter = "<<MaxIter<<endl;
    cout<<"Number of Shifts = "<<n_shift<<endl;
    cout<<"Tol = "<<tol<<endl;
    cout<<"Mass squared = "<<msqr<<endl;
    cout<<"lambda = "<<lambda<<endl;
    cout<<"Levels = "<<Levels<<endl;
    cout<<"Source Position = "<<src_pos<<endl;
    cout<<"Mass squared Correction = "<<C_msqr<<endl;
    cout<<"Lattice normalisation = "<<N_latt<<endl;
    cout<<"DiskScale = "<<DiskScale<<endl;
    cout<<"**********************************************************"<<endl<<endl;
  }
  
  void init(int argc, char **argv) 
  { 
    std::string BC(argv[1]);
    if (BC == "D" || BC == "d") 
	bc = true;
    else if (BC == "N" || BC == "n") 
	bc = false;
    else 
      {
	cout<<"Invalid boundary condition given. Use D/d for Dirichlet or N/n for Neumann."<<endl;
	exit(0);
      }

    std::string Centre(argv[2]);
    if (Centre == "V" || Centre == "v") 
      Vcentre = true;
    else if (Centre == "C" || Centre == "c") 
	Vcentre = false;
    else 
      {
	cout<<"Invalid centre condition given. Use V/v for Vertexcentred or C/c for Circumcentred."<<endl;
	exit(0);
      }

    std::string verbose(argv[3]);
    if (verbose == "V" || verbose == "v") 
	verbosity = true;
    else if(verbose == "Q" || verbose == "q") 
	verbosity = false;
    else 
      {
	cout<<"Invalid Verbosity conditions given. Use verbose/quiet"<<endl;
	exit(0);
      }

    MaxIter = atoi(argv[4]);
    tol     = atof(argv[5]);
    msqr    = atof(argv[6]);
    lambda  = atof(argv[7]);
    Levels  = atoi(argv[8]);
    src_pos = atoi(argv[9]);
    
    if(atof(argv[10]) == 0) C_msqr = -0.0126762/msqr + 0.0689398*msqr + 2.02509;
    else C_msqr = atof(argv[10]);
    
    if(atof(argv[11]) == 0) N_latt = 0.294452/(msqr + 0.766901) + 0.0788137;
    else N_latt = atof(argv[11]);
    
    q = atoi(argv[12]);
    n_shift = atoi(argv[13]);    
  }
};

class Vertex{
 public:

  double temporal_weight = 1.0;

  //If the pos value is -1, it is not connected
  //to the graph.
  int pos = -1;
  
  //Nearest neighbours for up to q=9 and 2 temporal directions.
  int nn[11] = {0,0,0,0,0,0,0,0,0,0,0};

  //How many forward links (important in the
  //buildGraph() function.
  int fwdLinks;

  //Positon on the Poincare disk.
  complex<Float> z;
  
  //Phi field value at this vertex.
  double phi = 0.0;
  
  //Old phi field value at this vertex (HMC).
  double phi_old = 0.0;
  
  //Ising field value at this vertex.
  int ising = 0.0;

};


typedef vector<Vertex> Graph;

complex<Float> T(complex<Float> z,  complex<Float> w);
complex<Float> R(complex<Float> z, complex<Float> omega);
complex<Float> flip(complex<Float> z, complex<Float> z1, complex<Float> z2);
Float s(complex<Float> z);
Float r(Float s );
Float d12(complex<Float> z1, complex<Float> z2);
Float sigma(complex<Float> z1, complex<Float> z2, float t);
Float s3p(int q);
Float area3q(int q);
Float areaGeneral(Param P, Float A, Float B, Float C);
Float centralRad(Float s);
complex<Float> DisktoUHP(complex<Float> z);
complex<Float> UHPtoDisk(complex<Float> u);
complex<Float> inversion(complex<Float> z0, Float r);
complex<Float> squareInversion(complex<Float>z0, Float r1, Float r2 );
Float greens2D(complex<Float> z, complex<Float> w);
Float greensM2D(complex<Float> z, complex<Float> w, Param p);
complex<Float> newVertex(complex<Float> z,complex<Float> z0,int k, int q);

void PrintNodeTables(const vector<Vertex> NodeList, Param P);

//- Edge length from center z = 0
Float edgeLength(int q) {
  return sqrt( 1 - 4*sin(M_PI/q)*sin(M_PI/q) );
}

//Using the formula c(n) = (q-4)*c(n-1) - c(n-2) where c is the
//number of nodes on circumference at level n, we can construct
//the address of the end node on a given level for triangulation q:
//EN(lev,q) = SUM c(n) n=0..level
long unsigned int endNode(int lev, Param P) { 
  
  int q = P.q;
  
  //This is the case where a vertex is at the centre of the disk
  if(P.Vcentre == true) {
    //Explicit results for level <= 2.
    if(lev==0) return 0;
    if(lev==1) return q;
    if(lev==2) return q*(q-4) + q;
    
    //level >= 3
    int p=q-4; //A convenience  
    long unsigned int c[20]; //Array to hold the circumference info
    
    //initialise array
    for(int i=0; i<20; i++) c[i] = 0;
    //manually set 0,1,2 entries:
    c[0] = 0;
    c[1] = q;
    c[2] = p*q;
    
    long unsigned int EN = 0; //address of node at end of level;  
    //Loop up to lev
    for(int n=3; n<lev+1; n++) {
      c[n] = p*c[n-1] - c[n-2];
      EN += c[n];
    }
    EN += (c[1] + c[2]);
    return EN;
  }

  //This is the case where a circumcentre is at the centre of the disk.
  else {
    //Explicit results for level <= 1.
    if(lev==0) return 2;
    if(lev==1) return (q-3)*3 + 2;
    
    //level >= 2
    int p=q-4; //A convenience  
    long unsigned int c[20]; //Array to hold the circumference info
    
    //initialise array
    for(int i=0; i<20; i++) c[i] = 0;
    //manually set 0,1,2 entries:
    //NB!!! There are three nodes on level 0, but the END NODE is addressed
    //as 2. Therefore, when correcting the number of nodes on a
    //circumference, we use 3 (there are three nodes)
    //but when giving the endNode count, we use 2 (we count from 0)
    c[0] = 3;       
    c[1] = (q-3)*3;
    
    long unsigned int EN = 0; //address of node at end of level;  
    //Loop up to lev
    for(int n=2; n<lev+1; n++) {
      c[n] = p*c[n-1] - c[n-2];
      EN += c[n];
    }
    EN += (2 + c[1]); //0,1,2 nodes to add from circumference 0
    return EN;
  }
}

//- Get the z coordinates of every node on the Poincare disk 
void GetComplexPositions(Graph &NodeList, Param& P){

  int q = P.q;
  int Levels = P.Levels;
  
  if(P.Vcentre == true) {
    //Assume for now that the origin (level 0) is a vertex
    NodeList[0].z = 0.0;
    //Assert that node 1 is on the real axis
    complex<Float> init(edgeLength(q),0.0);
    NodeList[1].z = init;
    //Rotate to create level level 1
    for(int k=1; k<q+1; k++) {
      NodeList[k].z = newVertex(init, 0.0, k-1, q);
    }
    //For every node on level >=1, the zeroth
    //nn is the (n-1)th link on the same level. If
    //n=1, the node address is the endnode value.
    for(int l=1; l<Levels+1; l++) {
      for(long unsigned int n=endNode(l-1,P)+1; n<endNode(l,P)+1; n++) {
	for(int k=0; k<q; k++) {
	  if(NodeList[n].nn[k] != -1) {
	    NodeList[NodeList[n].nn[k]].z = newVertex(NodeList[NodeList[n].nn[0]].z, NodeList[n].z, k, q);
	    //NodeList[NodeList[n].nn[k]].temporal_weight = (1+pow(abs(NodeList[NodeList[n].nn[k]].z),2))/(1-pow(abs(NodeList[NodeList[n].nn[k]].z),2));
	    //NodeList[NodeList[n].nn[k]].temporal_weight = 1.0 / pow(cos(M_PI*abs(NodeList[NodeList[n].nn[k]].z)/2),1);
	  }
	}
      }
    }    
  }
  else {
    
    Float numer = sqrt(cos(M_PI*(q+6)/(6*q)) - sin(M_PI/q));
    Float denom = sqrt(sin(M_PI/q) + sin(M_PI*(q+3)/(3*q)));    
    Float init_mod = sqrt(norm(numer/denom));
    
    //Assume that node 0 lies on +ve real axis
    complex<Float> init_0(init_mod,0.0);
    NodeList[0].z = init_0;
    
    //Assert that node 1 is node 0 rotated by 2*PI/3
    complex<Float>init_1(init_mod*cos(2.0*M_PI/3.0),
			  init_mod*sin(2.0*M_PI/3.0));
    
    NodeList[1].z = init_1;
    //Rotate node 1 about node 0 to create level 0 (the equilateral triangle)
    NodeList[2].z = newVertex(init_1, init_0, 1, q);

    //For every node on level >=1, the zeroth
    //nn is the (n-1)th link on the same level. If
    //n=1, the node address is the endnode value.
    for(long unsigned int n=0; n<endNode(1,P)+1; n++) {
      for(int k=0; k<q; k++) {
	if(NodeList[n].nn[k] != -1) {
	  NodeList[NodeList[n].nn[k]].z = newVertex(NodeList[NodeList[n].nn[0]].z, NodeList[n].z, k, q);
	}
      }
    }
    for(int l=1; l<Levels+1; l++) {
      for(long unsigned int n=endNode(l-1,P)+1; n<endNode(l,P)+1; n++) {
	for(int k=0; k<q; k++) {
	  if(NodeList[n].nn[k] != -1) {
	    NodeList[NodeList[n].nn[k]].z = newVertex(NodeList[NodeList[n].nn[0]].z, NodeList[n].z, k, q);
	    //NodeList[NodeList[n].nn[k]].temporal_weight = (1+pow(abs(NodeList[NodeList[n].nn[k]].z),2))/(1-pow(abs(NodeList[NodeList[n].nn[k]].z),2));
	  }
	}
      }
    }
  }
}

//- For each node n, with a link to another node,
//  it checks that the neighbour table on the linked
//  node contains the original node n as a neighbour.
void ConnectivityCheck(Graph &NodeList, Param P){

  int q = P.q;
  int Levels = P.Levels;
  int TotNumber = (endNode(Levels,P)+1);
  
  //Object to hold boolean values of graph connectivity.
  vector<Vertex> AuxNodeList(TotNumber);
  //Initialise to 0.
  for(int n = 0; n <TotNumber;n++)
    for(int mu = 0; mu < q; mu++) {
      AuxNodeList[n].nn[mu] = 0;
    }
  
  for(long unsigned int n=0; n<TotNumber; n++) {
    for(int m=0; m<q; m++) {
      //Check that the link is valid
      if(NodeList[n].nn[m] != -1) {
	for(int p=0; p<q; p++) {
	  //Loop over all links on the linked node,
	  //check if original node exists in neighbour
	  //table.
	  if( n == NodeList[ NodeList[n].nn[m] ].nn[p] ) {
	    AuxNodeList[n].nn[m] = 1;
	  }
	}
      }
    }
  }

  if(P.verbosity) PrintNodeTables(AuxNodeList, P);
}

void PrintNodeTables(const vector<Vertex> NodeList, Param P) {

  int q = P.q;
  int Levels = P.Levels;
  
  cout << endl << "lev = " << 0 << endl;
  
  if(P.Vcentre) {
    cout << endl<< " Node number = " << 0 << " : ";
    for(int i = 0; i < q; i++) cout << NodeList[0].nn[i] << "  ";
  }      
  else {
    for(long unsigned int n = 0; n < endNode(0,P)+1; n++) {
      cout << endl<< " Node number = " << n << " FL="<<NodeList[n].fwdLinks<<" : ";
      for(int i = 0; i < q; i++) cout << NodeList[n].nn[i] << "  ";
    } 
  }
  for(int lev = 1; lev < Levels+1; lev++)  {
    cout << endl << "lev = " << lev << endl;
    for(long unsigned int n = endNode(lev-1,P)+1; n < endNode(lev,P)+1; n++) {
      cout << endl<< " Node number = " << n << " FL="<<NodeList[n].fwdLinks<<" : ";
      for(int i = 0; i < q; i++) cout << NodeList[n].nn[i] << "  ";
    }
  }   
  cout<<endl;
}

void PrintComplexPositions(const vector<Vertex> NodeList, Param P) {
  
  int Levels = P.Levels;
  
  if(P.verbosity) cout<<endl<<"#Printing for Level 0"<<endl;
  for(long unsigned int n=0; n<endNode(0,P)+1; n++) {
    if(P.verbosity) {
      cout<<"n="<<n<<" z="<<NodeList[n].z.real()<<","<<NodeList[n].z.imag();
      cout<<" |z|="<<abs(NodeList[n].z)<<" phi="<<arg(NodeList[n].z);
    }
  }
  for(int l=1; l<Levels+1; l++) {
    if(P.verbosity) cout<<endl<<"Printing for Level "<<l<<endl;
    for(long unsigned int n=endNode(l-1,P)+1; n<endNode(l,P)+1; n++) {
      if(P.verbosity) {
	cout<<"n="<<n<<" z="<<NodeList[n].z.real()<<","<<NodeList[n].z.imag();
	cout<<" |z|="<<abs(NodeList[n].z)<<" phi="<<arg(NodeList[n].z);
	cout<<endl;
      }
    }
  }
}

void CheckArea(const vector<Vertex> NodeList, Param P) {

  Float length_01 = 0.0;
  Float length_02 = 0.0;
  Float length_12 = 0.0;
  Float equi_area = area3q(P.q);
  Float ave       = 0.0;

  Float sig1 = 0.0;
  Float sig2 = 0.0;
  int count = 0;

  if(P.verbosity) cout<<endl<<"Checking boundary areas"<<endl;
  for(long unsigned int n=endNode(P.Levels-2,P) + 1; n<endNode(P.Levels-1,P)+1; n++) {
    for(int k=0; k<NodeList[n].fwdLinks; k++) {
      length_01 = d12(NodeList[n].z, NodeList[NodeList[n].nn[k+1]].z);
      length_02 = d12(NodeList[n].z, NodeList[NodeList[n].nn[k+2]].z);
      length_12 = d12(NodeList[NodeList[n].nn[k+1]].z, NodeList[NodeList[n].nn[k+2]].z);      
      ave += areaGeneral(P, length_01, length_02, length_12);
      count++;
    }
  }
  
  ave /= count;
  
  for(long unsigned int n=endNode(P.Levels-2,P) + 1; n<endNode(P.Levels-1,P)+1; n++) {
    for(int k=0; k<NodeList[n].fwdLinks; k++) {
      length_01 = d12(NodeList[n].z, NodeList[NodeList[n].nn[k+1]].z);
      length_02 = d12(NodeList[n].z, NodeList[NodeList[n].nn[k+2]].z);
      length_12 = d12(NodeList[NodeList[n].nn[k+1]].z, NodeList[NodeList[n].nn[k+2]].z);      
      if(P.verbosity) cout<<"n="<<n<<" area "<<k+1<<" = "<<areaGeneral(P, length_01, length_02, length_12)<<endl;
      sig1 += pow(equi_area - areaGeneral(P, length_01, length_02, length_12),2);
      sig2 += pow(ave       - areaGeneral(P, length_01, length_02, length_12),2);
    }
  }

  sig1 /= count - 1;
  sig2 /= count - 1;
  sig1 = sqrt(sig1);
  sig2 = sqrt(sig2);

  cout<<"Boundary areas"<<endl;
  cout<<"AREA EQUI = "<<equi_area<<endl;
  cout<<"AREA STD DEV W.R.T. EQUI = "<<sig1<<endl;
  cout<<"AREA AVE = "<<ave<<endl;  
  cout<<"AREA STD DEV W.R.T AVE = "<<sig2<<endl;  

}

void CheckEdgeLength(const vector<Vertex> NodeList, Param P) {
  
  int q = P.q;
  int Levels = P.Levels;
  Float length = 0.0;
  Float sig = 0.0;
  int  nn_node;
  bool Vcentre = P.Vcentre;
  Float length_0 = d12(NodeList[0].z, NodeList[1].z);
  Float tol = 1e-2;

  //Level 0 is specific to how the graph is centred.
  if(Vcentre) {
    if(P.verbosity) cout<<" lev =  " << 0 << endl;
    if(P.verbosity) cout<<endl<<" Node number = "<<0<<" : "<<endl;
    for(int i = 0; i < q; i++){
      nn_node = NodeList[0].nn[i];
      length = d12(NodeList[0].z, NodeList[nn_node].z);
      if(P.verbosity) cout<<" "<<NodeList[0].nn[i]<<" > "<<length<<"  ";
    }
  }
  else {
    if(P.verbosity) cout<<" lev = "<<0<<endl;
    if(P.verbosity) cout<<endl<<" Node number = "<<0<<" : "<<endl;
    for(int i = 0; i < 2; i++){
      nn_node = NodeList[0].nn[i];
      length = d12(NodeList[0].z, NodeList[nn_node].z);
      if(P.verbosity) cout << NodeList[0].nn[i] << " >  " << length<< "  ";
    }
  }
  
  for(int lev = 1; lev < Levels+1; lev++)  {
    if(P.verbosity) cout<<endl<<endl<<" lev = "<<lev<<endl;      
    for(long unsigned int n = endNode(lev-1,P) + 1;n < endNode(lev,P) + 1 ;n++) {
      if(P.verbosity) cout<<endl<<" Node number = "<<n<<":"<<endl;
      sig += pow( length_0 - d12(NodeList[n].z, NodeList[NodeList[n].nn[q-1]].z), 2);
      
      for(int i = 0; i <q; i++){
	nn_node = NodeList[n].nn[i];
	if(NodeList[n].nn[i] != -1 ) {
	  length = d12(NodeList[n].z, NodeList[nn_node].z);
	  if(P.verbosity) {
	    cout<<" to "<<NodeList[n].nn[i]<<" = "<<length<<" ";
	    if(abs(length - length_0)/length_0 > tol) cout<<"<-! "<<endl;
	    else cout<<"    "<<endl;
	  }
	}
      }
    }
  }
  sig /= endNode(Levels,P);
  sig = sqrt(sig);
  cout<<endl<<"LENGTH STD DEV = "<<sig<<endl;
  if(sig>tol) {
    cout<<"WARNING: Hypergeometric length STD_DEV has diverged over "<<tol<<endl;
    //exit(0);
  }
}


//Data file for lattice/analytical propagator data,
void Bulk2Bdry(vector<Vertex> NodeList, Float *phi, Param p) 
{
  long unsigned int TotNumber = (endNode(p.Levels,p) + 1);
  Float norm = 0.0;
  for(long unsigned int i = 0;i < TotNumber; i++) norm += phi[i]*phi[i];
  for(long unsigned int i = 0;i < TotNumber; i++) phi[i] /= sqrt(norm); 
  long unsigned int j = p.src_pos;

  Float theta = 0.0;
  Float delta = 0.5 + sqrt(0.25 + p.msqr);
  complex<Float> ratio;
  complex<Float> src = NodeList[j].z;

  //Loop over circumference levels
  for(int lev=0; lev<p.Levels; lev++) {
    /*  
	sprintf(p.fname, "PROP_q%d_Lev%d_T%d_msqr%.3Le_srct0_srcpos%d_sinkt%dLev%d_%s_%s.dat",
	p.q,
	p.Levels,
	p.t,
	(Float)p.msqr,
	p.src_pos,
	t,
	lev+1,
	p.bc == true ? "Dirichlet" : "Neumann",
	p.Vcentre == true ? "Vertex" : "Circum");
	FILE *fp1;
	fp1=fopen(p.fname, "w");
    */
    //Loop over H2 disk
    for(long unsigned int k = endNode(lev,p)+1; k < endNode(lev+1,p)+1; k++) {
      
      //Construct i
      int i = k;
      complex<Float> snk = NodeList[i].z;
      ratio = NodeList[i].z/NodeList[j].z;
      theta = atan2( ratio.imag() , ratio.real() );
      
      //Float geo_dist = d12(src,snk);
      Float r = abs(NodeList[i].z);
      Float r_p = abs(NodeList[j].z);
      //Float xi = (cosh(delta_t)*(1+r)*(1+r_p) - 4*r*r_p*cos(theta)) / ((1-r)*(1-r_p));
      //Float analytic_prop = log( exp(-delta*sigma(src,snk,delta_t)) /
      //(1 - exp(-2*sigma(src,snk,delta_t))));
      /*
	if( i != j )  {
	fprintf(fp1, "%d %d %.8Le %.8Le %.8Le %.8Le %.8Le %.8Le %.8Le \n",
	//1 Timeslice, 2 H2 pos
	t, i,
	
	//3 source/sink angle
	(Float)theta,
	
	//4 lattice prop
	(Float)p.N_latt*(Float)phi[i],
	
	//5 invariant
	1.0/xi,
	//(Float)( ((Float)1.0-abs(snk))*((Float)1.0-abs(src))/(pow(abs(snk - src),2))),
	
	//6 AdS2p1 formula
	(Float)(exp(-delta*sigma(src,snk,delta_t)) / (1 - exp(-2*sigma(src,snk,delta_t)))),
	
	//7 geodesic
	(Float)sigma(src,snk,delta_t),
	
	//8 analytic propagator
	analytic_prop,
	
	//9 d12
	geo_dist
	);
      */
    }
    
    //fclose(fp1);
  }
}



/********************************************
Basic Hyperbolic Algebra. 

Moebius Transforms 

Hyperbolic reflection for geodesic in UHP

Given Line: (x-a)^2 + y^2 = r^2 
Determined by x = a \pm r at y = 0
              x = a, y = r


 RL(a,r, u) = a + r^2/(conj(u) - a)

Preserves the line and swap a  to infty.

Map to Disc: z = D(u) =  (u -I)/(1 -I * u) with u = x + i y
Map to UHP   u = U(z) = (z + I)/(1 + I * z);

Find UHP circle that hits  +/- theta_0 on  Disc

|z - A|^2 = R^2  
|z|^2 - |z| |A| cos(theta) + |A|^2 = R^2
boundary 1 - |A| cos (theta) + |A|^2 = R^2
pick A = real. when a = 0 with map

Need 3 point to define the mobius. Circle to Circle. 

****************************************************************/

// Translate w to 0 
complex<Float> T(complex<Float> z,  complex<Float> w)
{ //translate w to 0
  return (z - w)/(z*conj(w) + (Float)1.0);
}

// Rotate at z = 0
complex<Float> R(complex<Float> z, complex<Float> omega)
{
  // rotate by omega = exp [i theta] about z= 0
  return omega*z;
 }

//Reflection z accross the z1 -- z2 line
complex<Float> flip(complex<Float> z, complex<Float> z1, complex<Float> z2)
{
  // reflection (or flip)  z across (z1,z2)
  return T( R( conj( T(z,z1) ), z2/conj(z2) ), -z1 );
}

//Geodesic from z = 0 to z
Float  s(complex<Float> z)
{
   return log(((Float)1.0+abs(z))/((Float)1.0-abs(z)));
}

//Geodesic distance s from origin
Float r(Float s)
{
  return tanh(s/2);
}

//Geodesic distance from z1 to z2
Float d12(complex<Float> z, complex<Float> w)
{
  return log ( (abs((Float)1.0-conj(z)*w) + abs(z-w))/(abs((Float)1.0-conj(z)*w) - abs(z-w)));
}

//Geodesic distance from z1,t1 to z2,t2
Float sigma(complex<Float> z, complex<Float> w, float delta_t) {

  Float theta = atan2( (w/z).imag() , (w/z).real() );
  Float r = abs(z);
  Float r_p = abs(w);  
  Float xi = (cosh(delta_t)*(1+r)*(1+r_p) - 4*r*r_p*cos(theta)) / ((1-r)*(1-r_p)); 

  Float temp = cosh(delta_t)*cosh(r)*cosh(r_p)-sinh(r)*sinh(r_p)*cos(theta);

  return acosh(temp);
  //return acosh(xi);
  
}

// length of arc q fold triangle to origin.
Float s3p(int q)
{  //vertex centered Arc lengeth
  return (Float)2.0*acosh((Float)1.0/sin(M_PI/(Float)q));
}

// Area equilateral triangle with angles 2 pi/q
Float area3q(int q)
{
  //pi - (3 * hyp_angle) = defect
  return M_PI - (Float)3.0*(2.0*M_PI/(Float)q);
}

// Area non-equilateral triangle with side length a,b,c
Float areaGeneral(Param P, Float a, Float b, Float c) {
  //pi - (A+B+C) = defect
  
  // use general cosine law:
  // cos(C) = (cosh(c) - cosh(a)cosh(b)) / sinh(a)sinh(b)
  Float C = acos( -(cosh(c) - cosh(a)*cosh(b)) / (sinh(a)*sinh(b)) );
  
  // use general sine law:
  // sin(A)/sinh(a) = sin(B)/sinh(b) = ...
  Float B = asin( sinh(b)*sin(C)/sinh(c) );
  Float A = asin( sinh(a)*sin(C)/sinh(c) );

  return M_PI - (A+B+C);
}

//
Float centralRad(Float s)
{
  return (sqrt( cosh(s/2.0) - (Float)1.0/4.0) - 0.5*sqrt(3.0))/sqrt(cosh(s/2.0) -(Float)1.0);
}

//
complex<Float> DisktoUHP(complex<Float> z)
{
  // z = -1 -i, 1, i maps to u =  -1, 0 1, infty
  return (z + I)/((Float)1.0 + I * z);
}
//
complex<Float> UHPtoDisk(complex<Float> u)
{
  // u = 0, 1, infty  maps to -1 -i , 1, i  
  return (u - I)/((Float)1.0 - I*u); 
}

//- Rotate z about z0 by 2*k*pi/q 
complex<Float> newVertex(complex<Float> z,complex<Float> z0, int k, int q) {

  complex<Float> w( 0.0, 2.0 * sin(k * M_PI/q) );
  complex<Float> a( cos(k*M_PI/q)*((Float)1.0 - norm(z0)), sin(k*M_PI/q)*((Float)1.0 + norm(z0)) ); 
  w = w*z0;
  
  //cout<<"New z = "<<-(a*z - w)/(conj(w)*z - conj(a))<<endl;
  return - (a*z - w)/(conj(w)*z - conj(a)); 
}


complex<Float> inversion(complex<Float> z0, Float r)
{
  // z_image conj(z0) = r^2
  return r*2/conj(z0);
}

complex<Float> squareInversion(complex<Float>z0,Float r1,Float r2 )
{
  return inversion(inversion(z0, r1),r2);
}

Float greens2D(complex<Float> z, complex<Float> w)
{
  return -log( tanh ( log ( (abs((Float)1.0-conj(z)*w) + abs(z-w))/(abs((Float)1.0-conj(z)*w) - abs(z-w)) )/2 ) );    
}

Float greensM2D(complex<Float> z, complex<Float> w, Param p)
{

  //First compute 2F1  

  Float delta = 0.5 + sqrt(0.25 + p.msqr);
  Float h = 1;
  Float result = 0.0;
  Float result_0 = 0.0;
  Float geo = exp(-2*d12(z,w));
  Float a,b,c;
  Float tol = 1e-10;
  int n=0;
  bool conv = false;

  while( !conv && n < 10000 ) {    
    result_0 = result;
    a = tgamma(delta + n)/tgamma(delta);
    b = tgamma(h + n)/tgamma(h);
    c = tgamma(delta+1-h + n)/tgamma(delta+1-h);
    result += ( (a*b) / (c*tgamma(n+1)) ) * pow(geo,n);
    if( abs(result_0 - result)/result_0 < tol ) conv = true;
    n++;
    if(n%10000 == 0) cout<<n<<" 2F1 iters "<<geo<<" "<<abs(result_0 - result)/result_0<<endl; 
  }
  
  //Then compute coefficient. 
  result *= pow(geo,delta/2) * tgamma(delta) / (2*pow(M_PI,h)*tgamma(delta+1-h));

  return result;
}


/*
//D-function
Float DFunc(complex<Float> z, complex<Float> w)
{
  return ((z * conj(w))/(z-w)) * (2*PolyLog[2,z] - 2*PolyLog[2,w]+Log[z*w]*Log[(1-z)/(1-w)])
}
*/


/* (3,p)  packing lenght side = 2 ch^{-1} (1/2\sin (\pi/p)) 

cosh(side/2) = 1/(2 sin(\pi/p)) 

r_0=[ sqrt{ 4 cosh^2(side/2) -1} -(sqrt 3)/2]/ sqrt{cosh^2(side/2) -1}

*/


//Find the minimum value of an array of floats
float GetMinBoundaryRadius(vector<Vertex> NodeList, Param p)
{
  int Levels = p.Levels;
  int size = endNode(Levels,p)-endNode(Levels-1,p);   //number of nodes in outer level
  float outerCoords[endNode(Levels,p)-endNode(Levels-1,p)];
  for(long unsigned int i=endNode(Levels-1,p)+1; i<endNode(Levels,p)+1; i++)
    {
      outerCoords[i-endNode(Levels-1,p)-1] = abs(NodeList[i].z);
    }
  float minValue = 1;
  for (int i = 0; i < size; i++) 
    {
      //Find minValue
      if(outerCoords[i] != 0)
	{
	  if (outerCoords[i] < minValue)
	    {
	      minValue = outerCoords[i];
	    }
	}
    }
  return minValue;
}

//Find the maximum value of an array of floats
float GetMaxBoundaryRadius(vector<Vertex> NodeList, Param p)
{
  int Levels = p.Levels;
  int size = endNode(Levels,p)-endNode(Levels-1,p);   //number of nodes in outer level
  float outerCoords[endNode(Levels,p)-endNode(Levels-1,p)];
  for(long unsigned int i=endNode(Levels-1,p)+1; i<endNode(Levels,p)+1; i++)
    {
      outerCoords[i-endNode(Levels-1,p)-1] = abs(NodeList[i].z);
    }
  float maxValue = outerCoords[0];
  for (int i = 0; i < size; i++) 
    {
      //Find maxValue 
      if (outerCoords[i] > maxValue) 
	{
	  maxValue = outerCoords[i];
	}
    }
  return maxValue;
}

//Find the avg value of an array of floats
float GetAvgBoundaryRadius(vector<Vertex> NodeList, Param p)
{
  int Levels = p.Levels;
  int size = endNode(Levels,p)-endNode(Levels-1,p);   //number of nodes in outer level
  float outerCoords[endNode(Levels,p)-endNode(Levels-1,p)];
  for(long unsigned int i=endNode(Levels-1,p)+1; i<endNode(Levels,p)+1; i++)
    {
      outerCoords[i-endNode(Levels-1,p)-1] = abs(NodeList[i].z);
    }
  Float Value = 0.0;
  for (int i = 0; i < size; i++) 
    {
      Value += outerCoords[i];
    }
  return Value/size;
}



/* //Data file for lattice/analytical propagator data, */
/* void FourPoint(vector<Vertex> NodeList, Float *phi, Param p, int src_number, int *src_array, int src_position3)  */
/* { */
/*   int src_pos3 = src_position3; */
/*   int src_num = src_number; */
/*   long unsigned int TotNumber = (endNode(p.Levels,p) + 1) * p.t; */
/*   //int outer_num = endNode(p.Levels,p)-endNode(p.Levels-1,p); */

/*   int src_pos4 = src_array[3]; */
/*   int src_pos2 = src_array[1]; */
/*   int src_pos1 = src_array[0]; */
/*   //int src_pos2 = src_array[1]; */

/*   //Normalize the phi */
/*   Float norm = 0.0; */
/*   for(long unsigned int i = 0;i < TotNumber; i++) norm += phi[i]*phi[i]; */
/*   for(long unsigned int i = 0;i < TotNumber; i++) phi[i] /= sqrt(norm);  */
  
/*   //long unsigned int j = p.src_pos; */
/*   long unsigned int j = src_pos3; */

/*   //int T = p.t; */
/*   int T_offset = 0; */
/*   Float theta = 0.0; */
/*   Float delta = p.t < 2 ? 0.5 + sqrt(0.25 + p.msqr) : 1.0 + sqrt(1 + p.msqr); */
/*   complex<Float> ratio; */
/*   complex<Float> src = NodeList[j].z; */
 
/*   //Loop over timeslices */
/*   //for(int t=0; t<T; t++)  */
/*   int t=0; */
/*   // { */
/*       T_offset = (endNode(p.Levels,p) + 1) * t; */

/*       //Loop over circumference levels */
/*       //for(int lev=0; lev<p.Levels; lev++) */
/*       //for(int src_pos2 = src_pos3 + 1; src_pos2 < src_pos1; src_pos2++) */
/*       //	{ */
/* 	  sprintf(p.fname, "q%d_Lev%d_T%d_msqr%.3Le_srct0_srcpos%d_srcnum%d_sinkt%dLev%d_%s_%s.dat", */
/* 		  p.q, */
/* 		  p.Levels, */
/* 		  p.t, */
/* 		  (Float)p.msqr, */
/* 		  src_pos3, */
/* 		  src_num, */
/* 		  t, */
/* 		  p.Levels+1, */
/* 		  p.bc == true ? "Dirichlet" : "Neumann", */
/* 		  p.Vcentre == true ? "Vertex" : "Circum"); */
/* 	  FILE *fp1; */
/* 	  fp1=fopen(p.fname, "w"); */
      
/* 	  //Float prop = 1.0; */

/* 	  //Loop over H2 disk */
/* 	  for(long unsigned int k = endNode(0,p)+1; k < endNode(p.Levels,p)+1; k++)  */
/* 	    { */
/* 	      //Construct i - sink index */
/* 	      int i = k + T_offset; */
	      
/* 	      ratio = NodeList[i].z/NodeList[j].z; */
/* 	      theta = atan2( ratio.imag() , ratio.real() ); */
/* 	      complex<Float> snk = NodeList[i].z;   //location of current sink (endpoint) */
	      
/* 	      //index divided by disk size, using the int floor feature/bug, */
/* 	      //gives the timeslice for each index. */
/* 	      int t1 = j / (TotNumber/p.t); */
/* 	      int t2 = i / (TotNumber/p.t); */
/* 	      //Assume PBC. */
/* 	      int delta_t = (t2-t1) > p.t/2 ? (t2-t1) - p.t : (t2-t1); */
	      
/* 	      NodeList[j].z; */

/* 	      Float r = abs(NodeList[i].z);   //Radius of source */
/* 	      Float r_p = abs(NodeList[j].z); //Radius of sink */
/* 	      Float xi = (cosh(delta_t)*(1+r)*(1+r_p) - 4*r*r_p*cos(theta)) / ((1-r)*(1-r_p)); */
	  
/* 	      Float analytic_prop = log( exp(-delta*sigma(src,snk,delta_t)) / */
/* 				      (1 - exp(-2*sigma(src,snk,delta_t)))); */
    
/* 	      //cout<<"src_pos1 is: "<<src_pos1<<" src_pos 2 is: "<<src_pos2<<" src_pos3 is: "<<src_pos3<<" src_pos4 is: "<<src_pos4<<"\n"; */

/* 	      Float xr12 = NodeList[src_pos1].z.real()-NodeList[src_pos2].z.real(); */
/* 	      Float xr34 = NodeList[src_pos3].z.real()-NodeList[src_pos4].z.real(); */
/*               Float xr13 = NodeList[src_pos1].z.real()-NodeList[src_pos3].z.real(); */
/*               Float xr24 = NodeList[src_pos2].z.real()-NodeList[src_pos4].z.real(); */
/* 	      Float xi12 = NodeList[src_pos1].z.imag()-NodeList[src_pos2].z.imag(); */
/* 	      Float xi34 = NodeList[src_pos3].z.imag()-NodeList[src_pos4].z.imag(); */
/* 	      Float xi13 = NodeList[src_pos1].z.imag()-NodeList[src_pos3].z.imag(); */
/* 	      Float xi24 = NodeList[src_pos2].z.imag()-NodeList[src_pos4].z.imag(); */

/* 	      Float u_cr = ((xr12*xr12+xi12*xi12)*(xr34*xr34+xi34*xi34))/((xr13*xr13+xi13*xi13)*(xr24*xr24+xi24*xi24)); */
/* 	      Float Dfunction = (-2*log(1- sqrt(u_cr)))/(sqrt(u_cr)) + log(u_cr)/(sqrt(u_cr)-1); */
	      
	      
	      
	      
/* 	      //cout<<"Before IF. src_num is: "<<src_num<<". src_array[src_num] is: "<<src_array[src_num]<<"\n"; */
/* 		if( (i != src_pos1) && (i != src_pos3) && (i != src_pos4) && (i != src_pos2))   */
/* 		{ */
/* 		  fprintf(fp1, "%d %d %.8Le %.8Le %.8Le %.8Le %.8Le %.8Le %.8Le %.8Le %.8Le %.8Le %.8Le %d %.8Le \n", */
/* 			  //1 Timeslice, 2 H2 pos */
/* 			  t, i, */
			  
/* 			  //3 source/sink angle */
/* 			  (Float)theta, */
			  
/* 			  //4 lattice prop */
/* 			  (Float)p.N_latt*(Float)phi[i], */
			  
/* 			  //5 invariant */
/* 			  1.0/xi, */
/* 			  //(Float)( ((Float)1.0-abs(snk))*((Float)1.0-abs(src))/(pow(abs(snk - src),2))), */
			  
/* 			  //6 AdS2p1 formula */
/* 			  (Float)(exp(-delta*sigma(src,snk,delta_t)) / (1 - exp(-2*sigma(src,snk,delta_t)))), */
			  
/* 			  //7 geodesic */
/* 			  (Float)sigma(src,snk,delta_t), */

/* 			  //8  */
/* 			  NodeList[i].z.real(), */

/* 			  //9  */
/* 			  NodeList[i].z.imag(), */

/* 			  //10 */
/* 			  NodeList[j].z.real(), */

/* 			  //11 */
/* 			  NodeList[j].z.imag(), */

/* 			  //12 u cross-ratio  */
/* 			  u_cr, */

/* 			  //13 D-function  */
/* 			  Dfunction, */
			  
/* 			  //14 j */
/* 			  src_array[src_num], */

/* 			  //15 Analytic Propagator */
/* 			  analytic_prop */
/* 			  ); */
/* 		} */
/* 	    } */
/* 	  //prop = prop*(Float)p.N_latt*(Float)phi[i]; */
/* 	  fclose(fp1); */
/* 	  //	} */
/* } */


 //-------------//
 // (H)MC Stuff //
 //-------------//


//If true, prints a bunch of numbers!
bool flags = false;

//Utilities
void copyLattice(vector<Vertex> &Lattice, bool new_to_old, Param p);
void zeroField(Float *field, Param p);
Float measH(vector<Vertex> Lattice, Float *mom, Param p);
void gaussReal(Float *mom, Param p);

//Routines
int hmc(vector<Vertex> &Lattice, Param p, int iter);
void forceU(Float *fU, vector<Vertex> &Lattice, Param p);
void update_mom(Float *mom, Float *fU, Param p, double dt);
void update_phi(vector<Vertex> &Lattice, Float *mom, Param p, double dt);
void trajectory(vector<Vertex> &Lattice, Float *mom, Param p);

//Heatbath
void heatbath(vector<Vertex> &Lattice, Param p, int iter);

Float dHAve = 0.0;
int accepted_metropolis = 0;

void heatbath(vector<Vertex> &Lattice, Param p, int iter) {
  
  double phi_new = 0.0;
  double phi_new_sq = 0.0;
  double phi = 0.0;
  double phi_sq = 0.0;
  double lambda = 0.25*p.lambda;
  double msqr   = 0.50*p.msqr;
  
  double deltaH = 0.0;
  
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
      //cout << "Accepted" << endl;
      accepted_metropolis++;
      Lattice[i].phi = phi_new;
    }
    else if ( drand48() < exp(-deltaH)) {
      //cout<< "Acepted" << endl;
      accepted_metropolis++;
      Lattice[i].phi = phi_new;
    }
  }
}

// Utilities
//----------------------------------------------------------------------------

//Computes the action of the field
Float measH(vector<Vertex> Lattice, Float *mom, Param p) {

  Float KE = 0.0;
  Float PE =0.0;
  Float Hmom = 0.0;
  Float phi_sq = 0.0;
  Float phi = 0.0;
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
void gaussReal(Float *field, Param p) {  
  double r, theta, sum;
  for(int i=0; i<p.AdSVol; i++) {
    r = sqrt(-2.0*log(drand48()));
    theta = 2*M_PI * drand48();
    field[i] = r*cos(theta);
  }
  
  return;
}

void copyLattice(vector<Vertex> &Lattice, bool new_to_old, Param p){
  
  if (new_to_old) {    
    for (int i=0; i<p.AdSVol; i++) Lattice[i].phi_old = Lattice[i].phi;
  } else {
    for (int i=0; i<p.AdSVol; i++) Lattice[i].phi = Lattice[i].phi_old;
  }
}

void zeroField(Float *field, Param p){
  
  for (int i=0; i<p.AdSVol; i++) field[i] = 0.0;
}


// HMC routines
//----------------------------------------------------------------------------

int hmc(vector<Vertex> &Lattice, Param p, int iter) {

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

void trajectory(vector<Vertex> &Lattice, Float *mom, Param p) {

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

void forceU(Float *fU, vector<Vertex> &Lattice, Param p) {

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

void update_mom(Float *mom, Float *fU, Param p, double dt) {

  for(int i=0; i<p.AdSVol; i++) mom[i] -= fU[i] * dt;
  
}

void update_phi(vector<Vertex> &Lattice, Float *mom, Param p, double dt) {
  
  for(int i=0; i<p.AdSVol; i++) Lattice[i].phi += mom[i] * dt;
  
}

Float *wave_eqn(vector<Vertex> &Lattice, Float *phirad, Param p)
{
  Float *lap_phi = new Float[p.AdSVol];
  for(int i=0; i<p.AdSVol; i++)
    lap_phi[i] = 0.0;
  for(int i=0; i<p.AdSVol; i++)
    {
      for(int q=0; q<p.q; q++)
	{
	  //cout<<"phirad[i] is: "<<phirad[i];
	  lap_phi[i] += phirad[Lattice[i].nn[q]] - phirad[i];
	}
    }
  
  /*
  ofstream wavefile;
  wavefile.open("wave_eqn1.dat");
  for(int i=0; i<p.latVol; i++)
    wavefile<<lap_phi[i]<<" "<<phirad[i]<<"\n";
  wavefile.close();
  */
  return lap_phi;
}


#endif
