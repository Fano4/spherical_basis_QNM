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

    R = 100 * random.random()
    # Test with two real numbers
    l = random.randint(0,6)
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

def test_sph_harmo_norm():
    n_test=50
    av_re_error = 0
    for _ in range(n_test):
        # Parameters of the function to be tested
        l = random.randint(0, 6)
        m = random.randint(-l, l)
        # Monte Carlo variable distributions
        nsample = int(4e5)
        dS = 4 * np.pi / (nsample)
        x = -1. + 2. * np.random.rand(nsample)
        y = -1. + 2. * np.random.rand(nsample)
        z = -1. + 2. * np.random.rand(nsample)
        rr = (x ** 2 + y ** 2 + z ** 2) ** 0.5
        t = np.arccos(z / rr)
        f = np.arctan2(y, x) + np.pi
        integrand = np.zeros(nsample, dtype=complex)
        for ii in range(nsample):
            integrand[ii] = mf.pYlm(l, m, t[ii], f[ii]) * (-1)**m * mf.pYlm(l, -m, t[ii], f[ii])
        val = np.sum(integrand) * dS
        ref = 1
        av_re_error = av_re_error + ((np.real(val) - np.real(ref)) / np.real(ref)) / n_test
        print(av_re_error)
    assert (np.abs(av_re_error) < 1e-2)


def test_spherical_wf():
    # Test that we obtain the same norm as prescribed by sph_bessel_overlap
    # Since this is a 3D integral, we use a Monte Carlo method
    n_test=20
    av_re_error = 0
    max_val = 0

    for i in range(n_test):
        # Parameters of the function to be tested

        R = 10 * random.random()
        l = random.randint(0, 6)
        m = random.randint(-l, l)
        freq = 0.05 + 0.95 * random.random()
        k = freq
        print("test ", R,l,m,k)

        # Monte Carlo variable distributions
        nsample = int(4e5)
        dV = ((4 * np.pi / 3) * R ** 3) / (nsample)
        r = R * (np.random.rand(nsample))**(1/3)
        cost = -1. + 2. * np.random.rand(nsample)
        t = np.arccos(cost)
        f = 2*np.pi * np.random.rand(nsample)

        # Test for normalization
        integrand = np.zeros(nsample,dtype=complex)
        for ii in range(nsample):
            integrand[ii] = mf.pspherical_wave_function(l, m, k * r[ii], t[ii], f[ii]) * \
                            mf.pspherical_wave_function(l, m, k * r[ii], t[ii], f[ii],trans=True)
        val = np.sum(integrand) * dV
        # Spherical harmonics are normalized, which implies the integral is equivalent to the radial integral
        ref = mf.psph_Bessel_ovlp(l, k, k, R)
        err = ((np.real(val) - np.real(ref)) / np.real(ref))
        av_re_error = av_re_error + err/n_test
        print(np.real(val) ,np.real(ref),err)
    assert(np.abs(av_re_error) < 1e-2)

