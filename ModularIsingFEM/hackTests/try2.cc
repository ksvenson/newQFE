/** 

The rediculous confusion of C and C++ for example.

https://stackoverflow.com/questions/10694255/cmath-vs-math-h-and-similar-c-prefixed-vs-h-extension-headers

See https://www.cplusplus.com/reference/

Old c libraries are genearl global defines in std: namespace 

**/

#include <cmath>
#include <cstdio> //
#include <string>
#include <stack>
#include <vector>
#include "rng.h"

#include <iostream> // To use cout etc over rides <cstdio>
//using namespace  std; // without std::cout etc.

int main( )
{
  
  std::cout << "HI" << std::endl;

return 0;
}
