# This script finds the QNM modes for a spherical nano-particle based on dx.doi.org/10.1364/JOSAB.30.001996 (Ref 1).
# The homogeneous spherical nano-particle is characterized by a radius R and a dielectric constant eps.
# The dielectric constant is modelled using a Drude model. The QNM modes are computed in the basis of spherical wave
# functions evaluated at the mode frequency.  The coefficients of the expansion solve Maxwell's equations
# for the field inside the particles as an integral equation using the dyadic Green's function (Ref 1).

# The code works as follows. First, the eigenvalues and eigenvector for the QNM state are guessed.
# We then use a Rayleigh iterative scheme to find the eigenvectors and eigenvalues of the Green's function
# matrix in the spherical wave basis.
# The problem is deflated after having found each eigenvalue by orthonormalizing the
# induced field to the already found eigenvectors using Gram-Schmidt algorithm.

# The background Green's function is transformed in the spherical wave function basis by integrating the overlap matrix
# between the modes and the Green's function.

# We use a dimensionless unit system to simplify the problem (all constants = 1 ).
# The unit of length is chosen as 1um


import sys, os

sys.path.append(os.path.join(sys.path[0], '../src'))
print(sys.path)
import numpy as np
from functools import partial
from scipy import linalg as lg
import pickle

from src.spherical_basis_QNM.basis_set import basis_set
from src.spherical_basis_QNM.green_function import green_function
from src.spherical_basis_QNM.material import material
from src.spherical_basis_QNM.particle import particle
from src.spherical_basis_QNM.qnm_basis import qnm_basis
from src.spherical_basis_QNM.rayleigh_iteration import rayleigh_iteration as rit
from src.spherical_basis_QNM.plot_functions import plot_functions


def diagonal_term(part, sph_wf_basis, f):
    return np.eye(sph_wf_basis.size_ini) * (1 + (part.mat.eps(f) - part.med.eps(f)) / 3)


def greens_term(bg_green, part, f):
    return - part.med.k(f) ** 2 * part.mat.eps(f) * bg_green(f)


def nep_matrix(part, sph_wf_basis, bg_green, f):
    a = np.tensordot(diagonal_term(part, sph_wf_basis, f), np.eye(3), axes=0).swapaxes(1, 2).reshape(
        (3 * sph_wf_basis.size_ini, 3 * sph_wf_basis.size_ini))
    b = greens_term(bg_green, part, f).swapaxes(1, 2).reshape((3 * sph_wf_basis.size_ini, 3 * sph_wf_basis.size_ini))
    return a + b


def pseudo_spectrum(A: callable):
    y = np.linspace(0.25, 0, 100)
    x = np.linspace(1.4, 2.1, 200)
    func_to_plot = partial(abs_det_norm, A)
    plot_functions.plot_2d_func(func_to_plot, x, y, part='real', scale='log')
    # plot_functions.plot_func(func_to_plot,x)


def inv_det_norm(A, x, y):
    num = lg.det(A(x + y * 1j))
    # if num < 1e-4:
    #    num = 1
    return 1 / (num + 1e-6)


def abs_det_norm(A, x, y):
    num = np.abs(lg.det(A(x + y * 1j)))
    return num


if __name__ == '__main__':
    # Parameters of the computation:
    lmax = 0  # Maximum l-values for the solutions

    # We start with the nano-particle geometry and material parameters.
    R = 0.05  # 50 nm
    pos = np.array([0, 0, 0], dtype=float)  # centered at the origin
    mat_str = 'Au'
    mat = material.material(mat_str)  # Gold nanoparticle
    medium = material.material(1)  # Embedded in air
    part = particle.particle(pos, R, mat, medium)

    # Now the nano-particle response is computed by isolating the total field term in Eq. (8) of ref 1
    # We need to decompose the background dyadic Green's function in the spherical waves basis.  In that purpose,
    # we use Eqs (14), (15a) and (15b) of ref 1

    print("Initializing the basis set")
    sph_wf_basis = basis_set.basis_set(type="spherical", lmax=lmax, part=part)

    print("Initializing the Green's function")
    bg_green = green_function.green_function(sph_wf_basis)
    print("Green's function represented")

    reduced_nep = partial(nep_matrix, part, sph_wf_basis, bg_green)

    pseudo_spectrum(reduced_nep)

    exit()
    x0 = np.zeros(3 * sph_wf_basis.size_ini, dtype=complex)
    x0[0] = 1.
    z0 = [1.71 + 0.065j, 1.76 + 0.15j, 1.8 + 0.18j, 1.83 + 0.22j]
    eigen = rit.rayleigh_nep_solver(reduced_nep, x0, z0)
    eige = eigen[0]
    eigv = eigen[1]

    eige = np.array(eige)
    eigv = np.concatenate(eigv, axis=0)

    basis = qnm_basis.qnm_basis(sph_wf_basis, eigv, eige)

    filename = '../qnm_test_r_005_0.pkl'
    filehandler = open(filename, 'wb')
    pickle.dump(basis, filehandler)
    exit()
