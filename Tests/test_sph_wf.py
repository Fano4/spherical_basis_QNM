import sys
import random
import numpy as np
import scipy.integrate as integ
import scipy.special as sp

from functools import partial

sys.path.append("/Users/stephan/PycharmProjects/spherical_basis_QNM/")

import material
import particle
import spherical_wave_function as swf
from plot_functions import plot_func as plot


def wrapper_rotate_pos_args(fun, n, *args):
    args = list(args)
    newargs = args[-n:] + args[:-n]
    return fun(tuple(newargs))


def sph_wf_wrapper(l, f, part, r, **kwargs):
    k = part.k(f)
    fun = r ** 2 * sp.spherical_jn(l, k * r) ** 2
    if kwargs['mode'] == 'real':
        return np.real(fun)
    elif kwargs['mode'] == 'imag':
        return np.imag(fun)
    else:
        return fun


def test_init():
    a = np.linspace(0, 2, 10) + 0.3j
    l = np.arange(0, 10)
    m = np.arange(-5, 5)
    c = swf.sph_wf_symbol(a, l, m)

    assert (np.array_equal(c.a, a))
    assert (np.array_equal(c.l, l))
    assert (np.array_equal(c.m, m))


# Call on spherical wave functions are complicated to test, as they have analytical values that can only be computed
# using the very function we implemented. We will go for sanity checks, testing for the normalization
def test_call_normalization():
    accu_rel = 1e-5

    mat = material.material('Au')
    med = material.material(2.25)
    R = 0.05 * random.random()
    part = particle.particle(np.array((0, 0, 0)), R, mat, med)

    l = random.randint(0, 2)
    ml = random.randint(-l, l)
    sph_wf = swf.sph_wf_symbol(1, l, ml)

    f = 1.0

    resph_fun = partial(sph_wf_wrapper, l, f, part, mode='real')
    imsph_fun = partial(sph_wf_wrapper, l, f, part, mode='imag')

    ref = integ.quad(resph_fun, 0., R, epsabs=1e-15)[0] + 1j * integ.quad(imsph_fun, 0., R, epsabs=1e-15)[0]
    norm = 1 / sph_wf(np.array((0., 0., 0.)), f, part)[0] ** 2
    assert (np.abs(norm[0] - ref) / np.abs(ref) <= accu_rel)


def test_sph_deriv():
    l = 0
    ml = 0
    sph_wf = swf.sph_wf_symbol(1, l, ml)
    deriv = sph_wf.sph_deriv(0)
    print(deriv.a)
    print(deriv.l)
    print(deriv.m)
    assert (False)
