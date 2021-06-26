import sys
import scipy
import scipy.integrate as integ
import scipy.special as sp
import random
import numpy as np

sys.path.append('/Users/stephan/PycharmProjects/spherical_basis_QNM/')

import mathfunctions as mf


def test_sqr_root():
    z = 0.4 + 2j
    Rez = np.array([np.real(z ** 2)])
    Imz = np.array([np.imag(z ** 2)])
    mf.psquare_root(Rez, Imz)
    assert np.abs(Rez[0] + Imz[0] * 1j - z) < 1e-10


def reintegrand(r, k, kp, l):
    return np.real(r ** 2 * (sp.spherical_jn(l, k * r) * sp.spherical_jn(l, kp * r)))


def imintegrand(r, k, kp, l):
    return np.imag(r ** 2 * (sp.spherical_jn(l, k * r) * sp.spherical_jn(l, kp * r)))


def test_sph_bessel_overlap():
    """This is a test for the function that yields the overlap between two bessel functions with same l and distinct ks
    along a radial coordinate on a finite domain.  f(R,l,k,kp) = \\int_{0}^{R} r^2 dr j_{l}(k r) j_{l}(kp r)
    The k parameters can be defined on the complex plane"""

    R = 10
    # Test with two real numbers
    l = 2
    k = 1 + random.random()
    kp = 1 + random.random()

    result = integ.quad(lambda x: reintegrand(x, k, kp, l), 0, R)
    val = mf.psph_Bessel_ovlp(l, k, kp, R)

    assert np.abs(result[0] - val) < result[1]

    # Test with one real and one complex numbers
    k = 1 + random.random()
    kp = 1 + random.random() + random.random() * 1j

    reresult = integ.quad(lambda x: reintegrand(x, k, kp, l), 0, R)
    imresult = integ.quad(lambda x: imintegrand(x, k, kp, l), 0, R)
    val = mf.psph_Bessel_ovlp(l, k, kp, R)

    assert np.abs(reresult[0] + imresult[0] * 1j - val) < reresult[1] + imresult[1]

    # Test with two complex numbers
    k = 1 + random.random() + random.random() * 1j
    kp = 1 + random.random() + random.random() * 1j

    reresult = integ.quad(lambda x: reintegrand(x, k, kp, l), 0, R)
    imresult = integ.quad(lambda x: imintegrand(x, k, kp, l), 0, R)
    val = mf.psph_Bessel_ovlp(l, k, kp, R)

    assert np.abs(reresult[0] + 1j * imresult[0] - val) < reresult[1] + imresult[1]


def test_sph_harmo():

    t = np.pi * random.random()
    f = 2 * np.pi * random.random()
    assert (np.abs(sp.sph_harm(0, 0, f, t) - mf.pYlm(0,0, t, f)) < 1e-10)

    t = np.pi * random.random()
    f = 2 * np.pi * random.random()
    assert (np.abs(sp.sph_harm(0, 1, f, t) - mf.pYlm(1, 0, t, f)) < 1e-10)


    for _ in range(15):
        l = random.randint(0, 5)
        m = random.randint(-l, l)
        t = np.pi * random.random()
        f = 2 * np.pi * random.random()
        assert (np.abs(sp.sph_harm(m, l, f, t) - mf.pYlm(l, m, t, f)) < 1e-10)

