import random
import numpy as np
import scipy.integrate as integ
import scipy.special as sp
from functools import partial
from pytest import approx

from src.spherical_basis_QNM.basis_set import spherical_wave_function as swf
from src.spherical_basis_QNM.particle import particle
from src.spherical_basis_QNM.material import material


def test_check_values():
    # Check that the routine removes the zeros
    a = np.array([1., 0., 2., 0.])
    l = np.array([0, 1, 2, 3])
    ml = np.array([0, -1, 2, -2])

    refa = np.array([1., 2.])
    refl = np.array([0, 2])
    refml = np.array([0, 2])
    ref = [refa, refl, refml]
    val = swf.check_values(a, l, ml)
    assert (val[0] == ref[0]).all() and (val[1] == ref[1]).all() and (val[2] == ref[2]).all()

    # Check that the routine removes the invalid functions
    a = np.array([1., 2.5, 2.])
    l = np.array([1, -1, 2])
    ml = np.array([-1, 0, -4])

    refa = np.array([1.])
    refl = np.array([1])
    refml = np.array([-1])
    ref = [refa, refl, refml]
    val = swf.check_values(a, l, ml)
    assert (val[0] == ref[0]).all() and (val[1] == ref[1]).all() and (val[2] == ref[2]).all()

    # Check that the routine adds up identical functions
    a = np.array([1., 1., 1., 1.])
    l = np.array([0, 1, 1, 1])
    ml = np.array([0, -1, -1, -1])

    refa = np.array([1., 3.])
    refl = np.array([0, 1])
    refml = np.array([0, -1])
    ref = [refa, refl, refml]
    val = swf.check_values(a, l, ml)
    assert (val[0] == ref[0]).all() and (val[1] == ref[1]).all() and (val[2] == ref[2]).all()


def test_init():
    a = np.linspace(0, 2, 10) + 0.3j
    l = np.arange(0, 10)
    m = np.arange(-5, 5)
    checkval = swf.check_values(a, l, m)
    c = swf.sph_wf_symbol(a, l, m)

    assert (np.array_equal(c.a, checkval[0]))
    assert (np.array_equal(c.l, checkval[1]))
    assert (np.array_equal(c.m, checkval[2]))


def test_reset():
    a = np.array([1., 0., 2., 0.])
    l = np.array([0, 1, 2, 3])
    ml = np.array([0, -1, 2, -2])

    c = swf.sph_wf_symbol(a, l, ml)

    a = np.array([1., 2.5, 2.])
    l = np.array([1, -1, 2])
    ml = np.array([-1, 0, -4])
    checkval = swf.check_values(a, l, ml)
    c.reset(a, l, ml)

    assert (np.array_equal(c.a, checkval[0]))
    assert (np.array_equal(c.l, checkval[1]))
    assert (np.array_equal(c.m, checkval[2]))


# Call on spherical wave functions are complicated to test, as they have analytical values that can only be computed
# using the very function we implemented. We will go for sanity checks, testing for the normalization
def wrapper_rotate_pos_args(fun, n, *args):
    args = list(args)
    newargs = args[-n:] + args[:-n]
    return fun(tuple(newargs))


def sph_bessel_sqr_integrand_wrapper(l, f, part, r, **kwargs):
    k = part.k(f)
    fun = r ** 2 * sp.spherical_jn(l, k * r) ** 2
    if kwargs['mode'] == 'real':
        return np.real(fun)
    elif kwargs['mode'] == 'imag':
        return np.imag(fun)
    else:
        return fun


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

    resph_fun = partial(sph_bessel_sqr_integrand_wrapper, l, f, part, mode='real')
    imsph_fun = partial(sph_bessel_sqr_integrand_wrapper, l, f, part, mode='imag')

    ref = integ.quad(resph_fun, 0., R, epsabs=1e-15)[0] + 1j * integ.quad(imsph_fun, 0., R, epsabs=1e-15)[0]
    norm = 1. / sph_wf(np.array((0., 0., 0.)), f, part)[0][0] ** 2
    assert (np.abs(norm - ref) / np.abs(ref) <= accu_rel)


def test_sph_deriv():
    # Check simple case
    a = 1.5
    l = 0
    ml = 0
    sph_wf = swf.sph_wf_symbol(a, l, ml)
    res = sph_wf.sph_deriv(1)
    deriv = swf.sph_wf_symbol(res[0], res[1], res[2])
    assert approx(-np.sqrt(2. / 3.) * a) == deriv.a
    assert deriv.l == 1
    assert deriv.m == 1

    # Check case with l and ml != 0
    a = 2.345
    l = 2
    ml = -1
    sph_wf = swf.sph_wf_symbol(a, l, ml)
    res = sph_wf.sph_deriv(0)
    deriv = swf.sph_wf_symbol(res[0], res[1], res[2])
    assert approx(np.sqrt(8. / 35.) * a) == deriv.a[0]
    assert approx(-np.sqrt(1. / 5.) * a) == deriv.a[1]
    assert np.array_equal(deriv.l, np.array([3, 1]))
    assert np.array_equal(deriv.m, np.array([-1, -1]))


def sph_bessel_prod_integrand_wrapper(l, k1, k2, r, **kwargs):
    fun = r ** 2 * sp.spherical_jn(l, k1 * r) * sp.spherical_jn(l, k2 * r)
    if kwargs['mode'] == 'real':
        return np.real(fun)
    elif kwargs['mode'] == 'imag':
        return np.imag(fun)
    else:
        return fun


def test_med_sph_ovlp__sph_wf_ovlp():
    accu_rel = 1e-5

    mat = material.material('Au')
    med = material.material(2.25)
    R = 0.05 * random.random()
    part = particle.particle(np.array((0, 0, 0)), R, mat, med)

    l = random.randint(0, 2)
    ml = random.randint(-l, l)
    sph_wf = swf.sph_wf_symbol(1, l, ml)

    f = 0.1 + 0.9 * random.random()
    k1 = part.k(f)
    k2 = part.med.k(f)

    resph_fun = partial(sph_bessel_prod_integrand_wrapper, l, k1, k2, mode='real')
    imsph_fun = partial(sph_bessel_prod_integrand_wrapper, l, k1, k2, mode='imag')

    ref = integ.quad(resph_fun, 0., R, epsabs=1e-15)[0] + 1j * integ.quad(imsph_fun, 0., R, epsabs=1e-15)[0]

    # med_sph_wf_ovlp returns the overlap between normalized sph_wf.
    val = swf.med_sph_wf_ovlp(sph_wf, part, f)[0] / (sph_wf.norm(f, part, functype='mat')[0]
                                                     * sph_wf.norm(f, part, functype='background')[0])

    rel_err = np.abs(val - ref) / np.abs(ref)
    assert (rel_err <= accu_rel)

    f1 = 0.1 + 0.9 * random.random()
    f2 = 0.1 + 0.9 * random.random()
    k1 = part.k(f1)
    k2 = part.k(f2)

    resph_fun = partial(sph_bessel_prod_integrand_wrapper, l, k1, k2, mode='real')
    imsph_fun = partial(sph_bessel_prod_integrand_wrapper, l, k1, k2, mode='imag')

    ref = integ.quad(resph_fun, 0., R, epsabs=1e-15)[0] + 1j * integ.quad(imsph_fun, 0., R, epsabs=1e-15)[0]

    # sph_wf_ovlp returns the overlap between normalized sph_wf.
    val = swf.sph_wf_ovlp(sph_wf, part, f1, f2)[0] / (sph_wf.norm(f1, part, functype='mat')[0]
                                                      * sph_wf.norm(f2, part, functype='mat')[0])

    rel_err = np.abs(val - ref) / np.abs(ref)
    assert (rel_err <= accu_rel)


def sph_wf_outgo_fun_ovlp(l, k, kb, r):
    return r ** 2 * (sp.spherical_jn(l, kb * r) + 1j * sp.spherical_yn(l, kb * r)) * sp.spherical_jn(l, k * r)


def test_space_rad_integ():
    accu_rel = 1e-5

    mat = material.material('Au')
    med = material.material(2.25)
    R = 0.05 * random.random()
    part = particle.particle(np.array((0, 0, 0)), R, mat, med)

    f = 0.1 + 0.9 * random.random()
    k = np.real(part.k(f))
    kb = part.med.k(f)
    l = random.randint(0, 4)

    val = swf.space_rad_integ(l, k, kb)

    integrand = partial(sph_wf_outgo_fun_ovlp, l, k, kb)
    ref = integ.quad(integrand, 0., np.inf, epsabs=1e-15)[0]

    rel_err = np.abs(val - ref) / ref

    assert rel_err <= accu_rel
