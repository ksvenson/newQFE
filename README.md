# Quantum Finite Elements

Code for lattice simulations using the quantum finite element method.

We currently have Monte Carlo routines for ising and phi4 field theory,
including Metropolis and Wolff cluster updates, and overrelaxation for phi4.
The lattice can be defined on the following manifolds:

- Flat triangular lattice with periodic boundary conditions.
- AdS2 lattice with Dirichlet boundary conditions.

We try to follow the Google C++ Style Guide whenever it makes sense to do so:
https://google.github.io/styleguide/cppguide.html.
