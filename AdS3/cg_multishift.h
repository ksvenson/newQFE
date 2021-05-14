
#include <iostream>
#include <cmath>

using std::sqrt;
using std::abs;

void cg_multishift(Float **phi, Float *phi0, int n_shift, int size,
		   int resid_freq_check, int max_iter, Float eps,
		   Float* shifts, vector<Vertex> NodeList,
		   Param param, int freq_n);

// v1 = 0
void zero_vector(Float* v1, const int size)
{
  for (int i = 0; i < size; i++)
    v1[i] = 0.0;
}

// v1 = v2
void copy_vector(Float* v1, Float* v2, const int size)
{
  for (int i = 0; i < size; i++)
    v1[i] = v2[i];
}

// v2 += alpha v1
void caxpy(const Float alpha, Float* v1, Float* v2, const int size)
{
  for (int i = 0; i < size; i++)
    v2[i] += alpha*v1[i];
}

// v2 = v1 + alpha * v2
void cxpay(Float* v1, const Float alpha, Float* v2, const int size)
{
  for (int i = 0; i < size; i++)
    v2[i] = v1[i] + alpha*v2[i];
}

// v2 = alpha v1 + beta v2
void caxpby(const Float alpha, Float* v1, const Float beta, Float* v2, const int size)
{
  for (int i = 0; i < size; i++)
    v2[i] = alpha*v1[i] + beta*v2[i];
}

// v1 dot v2
Float dot(Float* v1, Float* v2, const int size)
{
  Float dv = 0.0;
  for (int i = 0; i < size; i++)
    dv += v1[i]*v2[i];

  return dv;
}

// ||v1||^2
Float norm2sq(Float* v1, const int size)
{
  Float dv = 0.0;
  for (int i = 0; i < size; i++)
    dv += v1[i]*v1[i];

  return dv;
}

// ||v1 - v2||^2
Float diffnorm2sq(Float* v1, Float* v2, const int size)
{
  Float dv = 0.0;
  for (int i = 0; i < size; i++)
    dv += (v1[i]-v2[i])*(v1[i]-v2[i]);

  return dv;
}

//Short term wrapper
void Minv_phi_ms(Float **phi, Float *phi0, vector<Vertex> NodeList, Param p, int freq_n){

  int n_shift = p.n_shift;
  int size = (endNode(p.Levels,p) + 1);
  int resid_freq_check = 10;
  int max_iter = p.MaxIter;
  Float eps = p.tol;
  Float *shifts = (Float*)malloc(n_shift*sizeof(Float));
  for(int i=0; i<n_shift; i++) shifts[i] = -0.2 + i*0.01;

  cg_multishift(phi, phi0, n_shift, size, resid_freq_check,
		max_iter, eps, shifts, NodeList, p, freq_n);
  
}

// Solves lhs = A^(-1) rhs using multishift CG as defined in
// http://arxiv.org/pdf/hep-lat/9612014.pdf
// Assumes there are n_shift values in "shifts".
// If they are sorted s.t. the smallest shift is the smallest
// (worst-conditioned solve), set worst_first = true. 
// resid_freq_check is how often to check the residual of other solutions.
// This lets us stop iterating on converged systems. 
void cg_multishift(Float **phi, Float *phi0, int n_shift, int size,
		   int resid_freq_check, int max_iter, Float eps,
		   Float* shifts, vector<Vertex> NodeList,
		   Param param, int freq_n) {
  
  // Initialize vectors.
  Float *r, *p, *Ap;
  Float **p_s;
  Float alpha, beta, beta_prev, rsq, rsqNew, bsqrt, tmp; 
  Float *alpha_s, *beta_s, *zeta_s, *zeta_s_prev;
  int k,i,n;
  int n_shift_rem = n_shift; // number of systems to still iterate on. 

  // Allocate memory.
  alpha_s = new Float[n_shift];
  beta_s = new Float[n_shift];
  zeta_s = new Float[n_shift];
  zeta_s_prev = new Float[n_shift];
  
  p_s = new Float*[n_shift];
  for (n = 0; n < n_shift; n++)
  {
    p_s[n] = new Float[size];
  }

  r = new Float[size];
  p = new Float[size];
  Ap = new Float[size];

  // Initialize values.
  rsq = 0.0; rsqNew = 0.0; bsqrt = 0.0; k=0;
  for (n = 0; n < n_shift; n++)
  {
    // beta_0, zeta_0, zeta_-1
    beta_s[n] = zeta_s[n] = zeta_s_prev[n] = 1.0;
    // alpha_0. 
    alpha_s[n] = 0.0;
  }
  beta = 1.0; alpha = 0.0;

  // Zero vectors;
  zero_vector(r, size); 
  zero_vector(p, size); zero_vector(Ap, size);
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // There can't be an initial guess... though it is sort of possible, in reference to:
  // http://arxiv.org/pdf/0810.1081v1.pdf
  
  // 1. x_sigma = 0, r = p_sigma = b.
  for (n = 0; n < n_shift; n++)
  {
    copy_vector(p_s[n], phi0, size);
    zero_vector(phi[n], size);
  }
  copy_vector(p, phi0, size);
  copy_vector(r, phi0, size);
  
  // Compute Ap.
  zero_vector(Ap, size);
  Mphi(Ap, p, NodeList, param, freq_n);
  
  // Compute rsq.
  rsq = norm2sq(r, size);

  // iterate till convergence
  for(k = 0; k< max_iter; k++) {
    
    // 2. beta_i = - rsq / pAp. Which is a weird switch from the normal notation, but whatever.
    beta_prev = beta; 
    beta = -rsq/dot(p, Ap, size);
    //cout << "beta = " << beta << "\n";
    
    for (n = 0; n < n_shift_rem; n++)
    {
      // 3. Calculate beta_i^sigma, zeta_i+1^sigma according to 2.42 to 2.44.
      // zeta_{i+1}^sigma = complicated...
      tmp = zeta_s[n]; // Save zeta_i to pop into the prev zeta.
      zeta_s[n] = (zeta_s[n]*zeta_s_prev[n]*beta_prev)/(beta*alpha*(zeta_s_prev[n]-zeta_s[n]) + zeta_s_prev[n]*beta_prev*(1.0-shifts[n]*beta));
      zeta_s_prev[n] = tmp; 
      
      //cout << "zeta_n = " << zeta_s[n] << ", zeta_{n-1} = " << zeta_s_prev[n];
      
      // beta_i^sigma = beta_i zeta_{n+1}^sigma / zeta_n^sigma
      beta_s[n] = beta*zeta_s[n]/zeta_s_prev[n];
      
      // 4. x_s = x_s - beta_s p_s
      caxpy(-beta_s[n], p_s[n], phi[n], size); 
      
      //cout << ", beta_n = " << beta_s[n] << "\n"; 
    }

    // 5. r = r + beta Ap
    caxpy(beta, Ap, r, size);
    
    // Exit if new residual is small enough
    rsqNew = norm2sq(r, size);

    // Comment this out to save your sanity.
    std::cout << "[CG-M-STATUS]: Iter " << k+1 << " RelTol " << sqrt(rsqNew)/bsqrt << "\n";

    // The residual of the shifted systems is zeta_s[n]*sqrt(rsqNew). Stop iterating on converged systems.
    if (k % resid_freq_check == 0)
    {
      for (n = 0; n < n_shift_rem; n++)
      {
        if (zeta_s[n]*sqrt(rsqNew) < eps*bsqrt) // if the residual of vector 'n' is sufficiently small...
        {
          n_shift_rem = n;
          break;
        }
      }
    }

    if ((abs(zeta_s[0])*sqrt(rsqNew) < eps*bsqrt) || n_shift_rem == 0 || k == max_iter-1)
    {
      break;
    }
    
    
  
    // 6. alpha = rsqNew / rsq.
    alpha = rsqNew / rsq;
    rsq = rsqNew; 
    
    //cout << "alpha = " << alpha << "\n";  
    
    for (n = 0; n < n_shift_rem; n++)
    {
      // 7. alpha_s = alpha * zeta_s * beta_s / (zeta_s_prev * beta)
      alpha_s[n] = alpha*zeta_s[n]*beta_s[n]/(zeta_s_prev[n] * beta);
      //cout << "alpha_n = " << alpha_s[n] << "\n";
      
      // 8. p_s = zeta_s_prev r + alpha_s p_s
      caxpby(zeta_s[n], r, alpha_s[n], p_s[n], size);
    }
    
    // Compute the new Ap.
    cxpay(r, alpha, p, size);
    Mphi(Ap, p, NodeList, param, freq_n);
  } 
    
  if(k == max_iter-1) {
    std::cout << "[CG-M-STATUS]: WARNING! Did not converge within maximum interations.\n";
  }

  k++;
  
  // Calculate explicit rsqs.
  for (n = 0; n < n_shift; n++) {
    zero_vector(Ap, size);
    Mphi(Ap, phi[n], NodeList, param, freq_n);
    caxpy(shifts[n], phi[n], Ap, size);
    std::cout << "[CG-M-STATUS]: Shift " << n << " ShiftVal " << shifts[n] << " RelTol " <<  sqrt(diffnorm2sq(Ap, phi0, size))/bsqrt << "\n";
  }
  
  
  // Free all the things!
  delete[] r;
  delete[] p;
  delete[] Ap;
  
  for (i = 0; i < n_shift; i++)
  {
    delete[] p_s[i];
  }
  delete[] p_s;
  
  delete[] alpha_s;
  delete[] beta_s;
  delete[] zeta_s;
  delete[] zeta_s_prev;

  std::cout << "[CG-M-STATUS]: Complete! Iter " << k << "\n";
}
