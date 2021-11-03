# This script finds the QNM modes for a spherical nano-particle based on dx.doi.org/10.1364/JOSAB.30.001996 (Ref 1).
# The homogeneous spherical nano-particle is characterized by a radius R and a dielectric constant eps.
# The dielectric constant is modelled using a Drude model. The QNM modes are computed in the basis of spherical wave
# functions at evaluated at the mode frequency.  The coefficients of the expansion solve Maxwell's equations
# for the field inside the particles as an integral equation using the dyadic Green's function.

# The code works as follows. First, the eigenvalues for the scattering state are guessed.
# The eigenvector is then guessed by computing the expansion coefficient for the field induced inside the nano-particle.
# We then use a Rayleigh iterative scheme to find the eigenvectors and eigenvalues of the background Green's function
# matrix in the spherical wave basis. The problem is reduced after having found each eigenvalue by orthonormalizing the
# induced field to the already found eigenvectors using Gram-Schmidt algorithm.

# The background Green's function is transformed in the spherical wave function basis by integrating the overlap matrix
# between the modes and the Green's function.

# We use a dimensionless unit system to simplify the problem (all constants = 1 )

import numpy as np

import basis_set
import green_function
import material
import field

# Parameters of the computation:
import particle

lmax = 2
wmin = 0.3
wmax = 0.8

guess_frq = 1 / 0.45

# We start with the nano-particle geometry and material parameters.
R = 1
pos = np.array([0, 0, 0], dtype=float)
mat_str = 'Au'
mat = material.material('Au')
medium = material.material(1.0)
part = particle.particle(pos, R, mat, medium)

# We set up an input field as a single background spherical wave
input_field_coeff = np.zeros((lmax + 1) ** 2, dtype=complex)
input_field_freq = np.zeros((lmax + 1) ** 2, dtype=float)
input_field_coeff[0] = 1
input_field_freq[0] = 1 / R
input_field = field.field(input_field_coeff, freq_cent=input_field_freq)

# Now the nano-particle response is computed by isolating the total field term in Eq. (8) of ref 1
# We need to decompose the background dyadic Green's function in the spherical waves basis.  In that purpose,
# we use Eqs (14), (15a) and (15b) of ref 1

sph_wf_basis = basis_set.basis_set(type="spherical", lmax=lmax, part=part)
bg_green = green_function.green_function()
exit()
