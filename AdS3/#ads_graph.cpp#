#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <vector>
#include <cstring>
#include <random>
#include <fstream>

using namespace std;
mt19937_64 rng(137);   //pseudo-random 64-bit number generator 
uniform_real_distribution<double> unif;
#define Float long double

#include <util.h>
#include <graph.h>
#include <cg.h>
#include <cg_multishift.h>
#include <eigen.h>

//void latticeScaling(vector<Vertex> &NodeList, Param& p);


// Begin Main Program
//====================================================
int main(int argc, char **argv) 
{
  Param p;    
  if(argc > 1) p.init(argc, argv);   
  
  //Number of boundary points
  int bdry_node_num = endNode(p.Levels,p)-endNode(p.Levels-1,p);
  cout<<"bdry_node_num is: "<<endNode(p.Levels,p)<<"-"<<endNode(p.Levels-1,p)<<"="<<bdry_node_num<<"\n";

  //Place source at first boundary point
  if(p.src_pos < 0) p.src_pos = endNode(p.Levels-1,p)+1;
  //p.src_pos=endNode(p.Levels,p);
  //p.src_pos = 43;
  //p.src_pos = 57;
  p.src_pos = 71;
  cout<<"Source position is: "<<p.src_pos<<"\n";
  cout<<"End Node is: "<<endNode(p.Levels,p)<<"\n";
  
  //Print graph endnode info
  //for(int i=1; i<20; i++) cout<<"Endnode("<<i<<") = "<<endNode(i,p)<<endl;

  int TotNumber = (endNode(p.Levels,p) + 1);
  vector<Vertex> NodeList(TotNumber);

  //Initialise. -1 in NodeList indicates that node is not yet populated.
  for(int n = 0; n <TotNumber;n++)
    for(int mu = 0; mu < p.q; mu++) { 
      NodeList[n].nn[mu] = -1;
    }

  //Construct neighbour table and z-coords
  BuildGraph(NodeList, p);           //populates NodeList[].nn[]
  
  GetComplexPositions(NodeList, p);  //populates NodeList[].z
  
  string bc = "";

  if(p.Vcentre)
    {
      bc = "V";
    }
  if(!p.Vcentre)
    {
      bc = "C";
    }

  /*** The below makes files that contain the z-coords and NN table ***/
  
  // ofstream myfile1;
  // myfile1.open("zcoords_m"+to_string(p.msqr)+"_L"+to_string(p.Levels)+"_q"+to_string(p.q)+"_bc"+bc+".dat");
  // for(int i=0; i<TotNumber; i++)
  //   {
  //     myfile1<<NodeList[i].z.real()<<" "<<NodeList[i].z.imag()<<endl;
  //   }
  // myfile1.close();

  // ofstream myfile2;
  // myfile2.open("links_m"+to_string(p.msqr)+"_L"+to_string(p.Levels)+"_q"+to_string(p.q)+"_bc"+bc+".dat");
  // for(int i=0; i<TotNumber; i++)
  //   {
  //     for(int k=0; k<p.q; k++)
  // 	{
  // 	  if(NodeList[i].nn[k] != -1)
  // 	    {
  // 	      myfile2<<i+1<<" "<<1+NodeList[i].nn[k]<<" "<<endl;
  // 	    }
  // 	}
  //   }
  // myfile2.close();


  cout.precision(17);
  cout<<"Average boundary radius for L="<<p.Levels<<" is: "<<GetAvgBoundaryRadius(NodeList, p)<<endl; 
  cout<<"Max bdry radius: "<<GetMaxBoundaryRadius(NodeList, p)<<endl;
  cout<<"Min bdry radius: "<<GetMinBoundaryRadius(NodeList, p)<<endl;
  
  /*** Below was used to facilitate comparison with Mathematica ***/
  
  // //Convention for theta=0: put source on farthest out point with smallest angle
  // complex<Float> pos = (0.0, 0.0);
  // int pos_num = 0;
  // float norm = abs(pos);
  // float angle = atan2( pos.imag(), pos.real() );
  // for(int i=endNode(p.Levels-1,p)+1; i<endNode(p.Levels,p)+1; i++)
  //   {
  //     float cur_norm = abs(NodeList[i].z);
  //     float cur_angle = atan2( NodeList[i].z.imag(), NodeList[i].z.real() );
  //     //cout<<"cur_norm is: "<<cur_norm<<endl;
  //     //cout<<"cur_angle is: "<<cur_angle<<endl;
  //     /*
  //     if(cur_norm == norm)
  // 	{
  // 	  if(cur_angle < angle)
  // 	    {
  // 	      pos = NodeList[i].z;
  // 	      pos_num = i;
  // 	      norm = cur_norm;
  // 	      angle = cur_angle;
  // 	    }
  // 	}
  //     */
  //     if(cur_norm > norm)
  // 	{
  // 	  pos = NodeList[i].z;
  // 	  pos_num = i;
  // 	  norm = cur_norm;
  // 	  angle = cur_angle;
  // 	}      
  //   }
  //cout<<"Old source position: "<<p.src_pos<<". New source position: "<<pos_num<<" with (norm, angle): ("<<norm<<", "<<angle<<")"<<endl;
  //p.src_pos = pos_num;

  //for(int i=endNode(p.Levels-1,p)+1; i<TotNumber; i++)
  //NodeList[i].z = NodeList[i].z/abs(NodeList[i].z);

  //Debug tools
  //FIXME: temporal stuff removed
  //ConnectivityCheck(NodeList, p);
  //CheckEdgeLength(NodeList, p);
  //CheckArea(NodeList, p);  

  if(p.verbosity) 
    {
      PrintNodeTables(NodeList, p);  
      PrintComplexPositions(NodeList, p);
    }
  
  if(p.src_pos < 0 || p.src_pos > TotNumber) 
    {
      cout<<"ERROR: Source Position must be g.e. 0 and l.e. "<<TotNumber-1;
      cout<<endl;
      exit(0);
    }
  
  //Lattice Scaling
  //latticeScaling(NodeList, p);   
  
  
  //-------------//
  // CG routines //
  //-------------//

  Float src_radius = abs(NodeList[p.src_pos].z);
  cout<<"src_radius is: "<<src_radius<<endl;
  Float src_rho = log((1+src_radius)/(1-src_radius));
  Float delta = 0.5*(1 + sqrt(1 + 4*p.msqr*p.msqr));
  cout<<"delta is: "<<delta<<endl;
  
  Float** G = new Float*[p.Lt+1];
  Float** phi0 = new Float*[p.Lt+1];
  Float** b0 = new Float*[p.Lt+1];
  float prop_norm = exp(-delta)/(1-exp(-2));
  cout<<"prop_norm is: "<<prop_norm<<endl;
  for(int freq_n = -28; freq_n <= 28; freq_n++)
    {
       
      float temp_time = 1.0*freq_n;
      float src_time_sigma = acosh(cosh(temp_time/57)*cosh(src_rho)*cosh(src_rho) - sinh(src_rho)*sinh(src_rho));
      float prop = exp(-delta*src_time_sigma)/(1-exp(-2*src_time_sigma));
      G[freq_n+28] = new Float[TotNumber];
      phi0[freq_n+28] = new Float[TotNumber];
      b0[freq_n+28] = new Float[TotNumber];
      for(int i=0; i<TotNumber; i++)
	{
	  G[freq_n+28][i] = 0.0;
	  phi0[freq_n+28][i] = 0.0;
	  b0[freq_n+28][i] = 0.0;
	}
      
      //b0[0][p.src_pos] = cos(2.0*M_PI*freq_n/p.Lt);
      //cout<<"prop is: "<<prop<<endl;

      //cout<<"prop/prop_norm is: "<<prop/prop_norm<<endl;
      Float temp_count = 0.0;
      for(int s=-28; s<=28; s++)
	{
	  
	  Float temp_time = 1.0*s;
	  Float src_time_sigma = acosh(1.0000001*cosh(temp_time/p.Lt)*cosh(src_rho)*cosh(src_rho) - sinh(src_rho)*sinh(src_rho));
	  Float prop = exp(-delta*src_time_sigma)/(1-exp(-2*src_time_sigma));
	  //cout<<"prop is: "<<prop<<endl;
	  temp_count += (1/sqrt(57.0))*(prop)*cos(2*M_PI*(freq_n-0.0)*(s-0.0)/57.0);
	  //cout<<"test is: "<<(1/sqrt(56.0))*(prop)*cos(2*M_PI*(freq_n-0)*(s-0)/56.0)<<endl;
	  //cout<<"temp_count in loop is: "<<temp_count<<" "<<freq_n<<" "<<s<<" "<<src_time_sigma<<" "<<acosh(1.0000001*cosh(temp_time/p.Lt)*cosh(src_rho)*cosh(src_rho) - sinh(src_rho)*sinh(src_rho))<<endl;
	  //cout<<"b0 is: "<<b0[freq_n][p.src_pos]<<endl;
	}
      //cout<<"cos(0) is: "<<cos(0/56.0)<<endl;
      b0[freq_n+28][p.src_pos] = temp_count;
      //cout<<"temp_count is: "<<temp_count<<endl;
      cout<<"b0["<<freq_n<<"]["<<p.src_pos<<"] is: "<<b0[freq_n+28][p.src_pos]<<" at radius: "<<src_rho<<endl;
      Float truesq1 = 0.0;
      //Float factor = cos(2.0*M_PI*freq_n/p.Lt);
      truesq1 = Minv_phi(phi0[freq_n+28], b0[freq_n+28], NodeList, p, freq_n);

    }
  //cout<<"G[0][0] is: "<<G[0][0]<<endl;
  //Print parameters prior to function call
  p.print();  
  //Mphi_ev(NodeList, p, freq_n);

  //Sum back over frequencies to get full Green's function


  
  for(int i=0; i<TotNumber; i++)
    {
      Float cur_rad = abs(NodeList[i].z);
      Float cur_rho = log((1+cur_rad)/(1-cur_rad));
      cout<<"cur_rho is: "<<cur_rho<<"for position: "<<i<<endl;
      for(int t=-28; t<=28; t++)
	{
	  Float temp1 = 0.0;
	  for(int w=-28; w<=28; w++)
	    {
	      temp1 += (1/sqrt(57.0))*(phi0[w+28][i])*cos(2*M_PI*w*t/57.0);
	    }
	  G[t+28][i] = temp1;
	  //for(int n=0; n<p.Lt; n++)
	  //{
	  //cout<<1.0/p.Lt<<"\n";
	  //G[t][i] += (1.0/sqrt(56.0))
	  //G[t][i] += (1.0/p.Lt)*exp(-2.0*M_PI*t/p.Lt)*(phi0[t][i]);
	  //}
	}
    }

    
  ofstream AdS3freqpropfile;
  AdS3freqpropfile.open("AdS3_freq_prop4.dat");
  for(int i=0; i<TotNumber; i++)
    {
      for(int t=0; t<p.Lt; t++)
	{
	  AdS3freqpropfile<<b0[t][i]<<" ";
	}
      AdS3freqpropfile<<"\n";
    }
  AdS3freqpropfile.close();

  
  //Print the AdS3 propagator to file
  
  ofstream AdS3propfile;
  AdS3propfile.open("AdS3_prop4.dat");
  for(int i=0; i<TotNumber; i++)
    {
      for(int t=0; t<p.Lt; t++)
	{
	  AdS3propfile<<G[t][i]<<" ";
	}
      AdS3propfile<<"\n";
    }
  AdS3propfile.close();
  
  //The below constructs the propagagtor from all-to-all by
  //inversion while cycling through where the "source" is.  
  /*
    Float** phi1 = new Float*[TotNumber];
    Float* b1 = new Float[TotNumber];
    for(int i=0; i<TotNumber; i++)
    {
    b1[i] = 0.0;
    phi1[i] = new Float[TotNumber];
    for(int j=0; j<TotNumber; j++)
    {
    phi1[i][j] = 0.0;
    }
    }

    for(int i=0; i<TotNumber; i++)
    {
    b1[i] = 1.0;
    Minv_phi(phi1[i], b1, NodeList, p);
    b1[i] = 0.0;
    }
  */

  //Example: print a file of the boundary-boundary correlator data 
  
   
  ofstream myfile;
  myfile.open("phi4.dat");
  for(int i=0; i<endNode(p.Levels,p)+1; i++)
    {
      complex<long double> ratio = NodeList[i].z/NodeList[p.src_pos].z;
      long double theta = atan2(ratio.imag(), ratio.real());
      myfile<<d12(NodeList[p.src_pos].z, NodeList[i].z)<<" "
		  <<theta<<" "
		  <<phi0[0][i]<<" "
		  <<abs(NodeList[i].z)<<" "
		  <<NodeList[i].z.real()<<" "
		  <<NodeList[i].z.imag()<<" "
		  <<NodeList[p.src_pos].z.real()<<" "
		  <<NodeList[p.src_pos].z.imag()<<" "
	    <<sigma(NodeList[p.src_pos].z, NodeList[i].z, 0)<<"\n";
    }
  myfile.close();
  
  //1 geodesic distance
  //2 theta
  //3 lattice prop
  //4 |z|
  //5 Re[z]
  //6 Im[z]
  //7 sigma

  //Example: print a file of the geodesic distance from all-to-all 

  ofstream distfile;
  distfile.open("sigma_m"+to_string(p.msqr)+"_L"+to_string(p.Levels)+"_q"+to_string(p.q)+"4.dat");
  for(int i=0; i<TotNumber; i++)
    {
      for(int j=0; j<TotNumber; j++)
	{
	  distfile << sigma(NodeList[i].z, NodeList[j].z, 0) << " ";
	}
      distfile << "\n";
    }
  distfile.close();

  
  ofstream phifile;
  phifile.open("phi_m"+to_string(p.msqr)+"_L"+to_string(p.Levels)+"_q"+to_string(p.q)+"4.dat");
  for(int i=0; i<TotNumber; i++)
    {
      for(int n=0; n<p.Lt; n++)
	{
	  phifile << phi0[n][i] << " ";
	}
      phifile << "\n";
    }
  phifile.close();

  
  ofstream timefile;
  timefile.open("time_m"+to_string(p.msqr)+"_L"+to_string(p.Levels)+"_q"+to_string(p.q)+"4.dat");
  for(int i=0; i<TotNumber; i++)
    {
      for(int t=-28; t<=28; t++)
	{
	  float temp_time = 1.0*t;
	  timefile << sigma(NodeList[i].z, NodeList[i].z, temp_time/p.Lt) << " ";
	}
      timefile << "\n";
    }
  timefile.close();

  ofstream rhofile;
  rhofile.open("rho_m"+to_string(p.msqr)+"_L"+to_string(p.Levels)+"_q"+to_string(p.q)+"4.dat");
  for(int i=0; i<TotNumber; i++)
    {
      Float temp2 = abs(NodeList[i].z);
      for(int t=0; t<p.Lt; t++)
	{
	  rhofile << log((1+temp2)/(1-temp2)) << " ";
	}
      rhofile << "\n";
    }
  rhofile.close();

  return 0;
}
