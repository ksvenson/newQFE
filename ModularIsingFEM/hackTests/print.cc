#include <cmath>
#include <cstdio>
#include <iostream> // Need for cout !
#include <string>
#include <stack>
#include <vector>
#include "rng.h"

using namespace  std;

int main( )
{
  int L = 4;
  int latVol = L*L;
    bool is_cluster[latVol];
    QfeRng rng(137);
    

     for(int x = 0; x < L ; x++)
       {  printf("  %d  \n", x);
	 cout << "   x = " << x << endl;
       }
	  
cout << "HI" << endl;

return 0;
}
