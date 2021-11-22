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
from functools import partial
import pickle

import basis_set
import green_function
import material
import particle
import qnm_basis
import rayleigh_iteration as rit
import plot_functions
import qnm_basis as qnm

# Parameters of the computation:


lmax = 0
wmin = 0.3
wmax = 0.8


def func(sph_wf_basis, l, ml, part, f, x, z):
    # def func(sph_wf_basis, x, z):
    r = np.array([x, 0, z])
    # return sph_wf_basis.part.inout(r)
    return sph_wf_basis.basis[sph_wf_basis.jlm_to_index(0, l, ml)](r, f, part, inout='out')[1]


def func_hess(sph_wf_basis, comp_1, comp_2, l, ml, part, f, x, z):
    # def func(sph_wf_basis, x, z):
    r = np.array([x, 0, z])
    # return sph_wf_basis.part.inout(r)
    return sph_wf_basis.basis_hessian[sph_wf_basis.jlm_to_index(0, l, ml)][comp_1][comp_2](r, f, part, inout='in')[1]


def diagonal_term(part, sph_wf_basis, f):
    return np.eye(sph_wf_basis.size_ini) * (1 + (part.mat.eps(f) - part.med.eps(f)) / 3)


def greens_term(bg_green, part, f):
    return - part.med.k(f) ** 2 * part.mat.eps(f) * bg_green(f)


def nep_matrix(part, sph_wf_basis, bg_green, f):
    a = np.tensordot(diagonal_term(part, sph_wf_basis, f), np.eye(3), axes=0).swapaxes(1, 2).reshape(
        (3 * sph_wf_basis.size_ini, 3 * sph_wf_basis.size_ini))
    b = greens_term(bg_green, part, f).swapaxes(1, 2).reshape((3 * sph_wf_basis.size_ini, 3 * sph_wf_basis.size_ini))
    return a + b


# We start with the nano-particle geometry and material parameters.
R = 0.05
pos = np.array([0, 0, 0], dtype=float)
mat_str = 'Au'
mat = material.material('Au')
medium = material.material(1.0)
part = particle.particle(pos, R, mat, medium)

f0 = 2.0727723391689494 - 2.3790001072293j

# Now the nano-particle response is computed by isolating the total field term in Eq. (8) of ref 1
# We need to decompose the background dyadic Green's function in the spherical waves basis.  In that purpose,
# we use Eqs (14), (15a) and (15b) of ref 1

print("Material dielectric constant : ")


def eps_fun(mat: material, x, y):
    f = x + 1j * y
    return mat.eps(f)


red_eps_fun = partial(eps_fun, mat)
x = np.linspace(1. / 1.5, 1 / 0.4, 250)
y = np.linspace(0, 0.1, 50)
# plot_functions.plot_2d_func(red_eps_fun, x, y)
plot_functions.plot_func(mat.eps, x)
plot_functions.plot_func(mat.n, x)

print("Initializing the basis set")
sph_wf_basis = basis_set.basis_set(type="spherical", lmax=lmax, part=part)

l = 1
ml = -1
xrange = np.linspace(- 20 * R, 20 * R, 400)
zrange = np.linspace(- 20 * R, 20 * R, 400)
X, Z = np.meshgrid(xrange, zrange)
Y = np.zeros(X.shape)
r = np.array([X, Y, Z])

print("Plotting a spherical wave function")
# red_fun = partial(func, sph_wf_basis, l, ml, part, f0)
# plot_functions.plot_2d_func(red_fun, xrange, zrange)
# exit()
# print("Plotting a spherical wave function Hessian component")
# red_fun = partial(func_hess, sph_wf_basis, 0, 0, l, ml, part, guess_frq)
# plot_functions.plot_2d_func(red_fun, xrange, zrange, part='real')

# print("Printing separation matrix projection")
# for i in range(sph_wf_basis.size):
#    for j in range(sph_wf_basis.size):
#        arr = sph_wf_basis.basis_separation_mat[i][j].sph_basis_proj(sph_wf_basis,guess_frq)
#        print(i,j)
#        print(arr)
# print("Initializing the Green's function")
bg_green = green_function.green_function(sph_wf_basis)

print("Calling Green's function")
array = bg_green(f0)
print("Green's function represented")

reduced_nep = partial(nep_matrix, part, sph_wf_basis, bg_green)

# rit.pseudo_spectrum(reduced_nep,10)

# exit()

x0 = np.zeros(3 * sph_wf_basis.size_ini, dtype=complex)
x0[0] = 1.
z0 = 1 / 0.7 - 0.005 * 1j
eigen = rit.rayleigh_nep_solver(reduced_nep, x0, z0)
eige = eigen[0]
eigv = eigen[1]

eige = np.array(eige)
eigv = np.concatenate(eigv, axis=0)

basis = qnm_basis.qnm_basis(sph_wf_basis, eigv, eige)

filename = 'qnm_test_r_005_0.pkl'
filehandler = open(filename, 'wb')
pickle.dump(basis, filehandler)
exit()
