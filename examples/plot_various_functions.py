import sys, os
from functools import partial
import numpy as np

sys.path.append(os.path.join(sys.path[0], '../src'))

from src.spherical_basis_QNM.material import material
from src.spherical_basis_QNM.particle import particle
from src.spherical_basis_QNM.basis_set import basis_set
from src.spherical_basis_QNM.plot_functions import plot_functions


def eps_fun(mat: material, x, y):
    f = x + 1j * y
    return mat.eps(f)


def func(sph_wf_basis, l, ml, part, f, x, z):
    r = np.array([x, 0, z])
    return sph_wf_basis.basis[sph_wf_basis.jlm_to_index(0, l, ml)](r, f, part, inout='inout')[1]


def func_hess(sph_wf_basis, comp_1, comp_2, l, ml, part, f, x, z):
    r = np.array([x, 0, z])
    return sph_wf_basis.basis_hessian[sph_wf_basis.jlm_to_index(0, l, ml)][comp_1][comp_2](r, f, part, inout='inout')[1]


def plot_dielectric_constant(mat_str):
    mat = material.material(mat_str)  # Gold nanoparticle
    print("Material dielectric constant : ")
    x = np.linspace(1. / 1.5, 1 / 0.3, 250)
    # plot_functions.plot_2d_func(red_eps_fun, x, y)
    plot_functions.plot_func(mat.eps, x)
    plot_functions.plot_func(mat.n, x)


def plot_qnm_bas_func(f0, R, mat_str, l, ml):
    # Set up particle geometry and material parameters
    mat = material.material(mat_str)  # Gold nanoparticle
    medium = material.material(2.25)  # Embedded in glass
    pos = np.array([0, 0, 0], dtype=float)  # centered at the origin
    part = particle.particle(pos, R, mat, medium)
    sph_wf_basis = basis_set.basis_set(type="spherical", lmax=l, part=part)

    xrange = np.linspace(- 20 * R, 20 * R, 200)
    zrange = np.linspace(- 20 * R, 20 * R, 200)
    X, Z = np.meshgrid(xrange, zrange)
    Y = np.zeros(X.shape)

    print("Plotting a spherical wave function")
    red_fun = partial(func, sph_wf_basis, l, ml, part, f0)
    plot_functions.plot_2d_func(red_fun, xrange, zrange)

    print("Plotting a spherical wave function Hessian component")
    red_fun = partial(func_hess, sph_wf_basis, 0, 0, l, ml, part, f0)
    plot_functions.plot_2d_func(red_fun, xrange, zrange, part='real')

    print("Printing separation matrix projection")
    for i in range(sph_wf_basis.size):
        for j in range(sph_wf_basis.size):
            arr = sph_wf_basis.basis_separation_mat[i][j].sph_basis_proj(sph_wf_basis, f0)
            print(i, j)
            print(arr)


if __name__ == '__main__':
    plot_dielectric_constant('Au')
    plot_qnm_bas_func(2.5, 5e-2, 'Au', 0, 0)
