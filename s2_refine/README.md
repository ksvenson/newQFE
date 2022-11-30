# s2_refine

Tools for generating lattices for a discretized 2-sphere S^2.

## s2_std_refine

Generates a standard S^2 lattice by triangulating the faces of an octahedron
or icosahedron. Usage is

`s2_std_refine <k> <q>`

where `<k>` is the refinement level and `<q>` is the base polytope type,
4 for an octahedron and 5 for and icosahedron.
Default is k=3 and q=5. The lattice will be saved to
`s2_std/q5k3_std.dat` and the orbit data will be saved to
`s2_std/q5k3_orbit.dat`. The orbit data is needed by `s2_std_uniform`. You
will need to create the directory `s2_std` before running the program.

## s2_lap_dec

Calculates the finite element weights for each simplex in the lattice using
the discrete exterior calculus method. Then the spectrum of the Laplacian
operator is calculated in the spherical harmonic basis. The site weights are
also used to measure the effectiveness of the site weights as an integration
measure. Usage is

`s2_lap_dec <base_path>`

where `base_path` is the base path for the lattice (without an extension). For
example, to analyze `s2_std/q5k3_std.dat`, `base_path` would be
`s2_std/q5k3_std`, and the output files will be

- `s2_std/q5k3_std_dec.dat`: lattice file with DEC weights
- `s2_std/q5k3_std_dec_lap.dat`: eigenvalues of Laplacian operator
- `s2_std/q5k3_std_dec_int.dat`: integrator values

## s2_std_uniform

Moves the lattice points to make the effective lattice spacing as uniform as
possible. The program does this by using a non-linear solver to minimize the
combined variance of 3 different measures of lattice uniformity.

- Site dual area
- Face area
- Face circumradius

It's unclear if any one of these is more important than the others in terms
of restoring rotational symmetry in a strongly coupled theory in the continuum
limit. This program minimizes the sum of the normalized variance of all 3
measures. Usage is

`s2_std_uniform <base_path>`

where `base_path` is the path to a "std" lattice and orbit file from
`s2_std_refine`, e.g. `s2_std/q5k3`. The output file will be
`s2_std/q5k3_uniform.dat`. Note that the DEC element weights will automatically
be calculated for the uniform lattice.

## s2_ct and s2xr_ct

Computes couterterms for S^2 and S^2 cross R lattices. Example usage is

`s2[xr]_ct --lattice_path s2_std/q5k3_uniform.dat --ct_path s2_std/q5k3_uniform_ct.dat`

The first option is the full path to the lattice file and the second option
is the path for the output file with the counterterms.
