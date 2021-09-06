/** 

The rediculous confusion of C and C++ for example.

https://stackoverflow.com/questions/10694255/cmath-vs-math-h-and-similar-c-prefixed-vs-h-extension-headers

See https://www.cplusplus.com/reference/

Old c libraries are genearl global defines in std: namespace 

**/

#include <cmath>
#include <cstdio>
#include <string>
#include <stack>
#include <vector>
#include "rng.h"

#include <iostream> // To use cout etc over rides <cstdio>
using namespace  std; // without std::cout etc.

int main( )
{
  int L = 4;
  int latVol = L*L;
    bool is_cluster[latVol];
    QfeRng rng(137);
    
   
     for(int x = 0; x < L ; x++){
       printf("  %d  \n", x);
       cout << x << endl;
     }
    	  
  for(int i =0 ; i< L;i++) {
    is_cluster[i] = true;
    printf(" is_cluster %d \n",is_cluster[i]);
  }

  for(int i =0 ; i< latVol;i++) {
    printf(" rng.RandReal() = %.12f \n",  rng.RandReal());

    printf(" rng.RandInt(0, p.latVol - 1) = %d \n",rng.RandInt(0, latVol - 1)); 
  }
  
  int pi = 31;

  // printf(" pi =   %d  latVol =  %d  pi/latVol =  %d double(pi)/latVol =  %.12f  pi/double(latVol) =  %.12f\n", pi, latVol, pi/latVol,double(pi)/latVol, pi/double(latVol));
  
  
cout << "HI" << endl;

return 0;
}
