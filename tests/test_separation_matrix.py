import numpy as np
from numpy import random as rd
import pytest
from src.spherical_basis_QNM.basis_set import spherical_wave_function as swf
from src.spherical_basis_QNM.separation_matrix import separation_matrix as sep
from src.spherical_basis_QNM.particle import particle
from src.spherical_basis_QNM.material import material


def test_special_case_0():
    # This function tests the call function of the separation matrix object
    # by asserting Eq (i) p 88 in Multiple scattering Interaction of time-harmonic waves with N obstacles
    # by P. A. Martin

    r = np.array([0.0, 0.0, 0.0])
    f = 1.0
    rad = 1.0

    mat = material.material(1.0)
    part = particle.particle(np.array([0.0, 0.0, 0.0]), rad, mat)

    n = 0
    m = 0
    N = 0
    M = 0
    print(" Testing with different parameters")
    while n == N and m == M:
        n = rd.randint(0, high=5)
        m = rd.randint(-n, high=n + 1)

        N = rd.randint(0, high=5)
        M = rd.randint(-N, high=N + 1)
    print(n, m, N, M)

    sepmat1 = sep.separation_matrix(1.0, n, m, N, M)

    assert sepmat1(r, f, part, type='in') == pytest.approx(0)

    print(" Testing with same parameters")
    n = rd.randint(0, high=5)
    m = rd.randint(-n, high=n + 1)
    N = n
    M = m
    print(n, m, N, M)

    sepmat1.reset_sepmat(1.0, n, m, N, M)
    assert sepmat1(r, f, part, type='in') == pytest.approx(1)


def test_special_case_1():
    # This function tests the call function of the separation matrix object
    # by asserting Eq (ii) p 88 in Multiple scattering Interaction of time-harmonic waves with N obstacles
    # by P. A. Martin
    r1 = np.array([0., 0., 0.])
    r2 = np.array([-1.45, 0.134, -0.765])
    b = r2 - r1

    f = 1.0
    rad = 2 * np.sum(r2 ** 2) ** 0.5

    n = 2
    m = 1

    mat = material.material(1.0)
    part = particle.particle(np.array([0.0, 0.0, 0.0]), rad, mat)
    wf1 = swf.sph_wf_symbol(1, n, m)
    ref = wf1(b, f, part)[1][0]
    sepmat = sep.separation_matrix(1.0, n, m, 0, 0)
    val = sepmat(b, f, part, type='in')

    assert val == pytest.approx(ref)


def test_identity():
    # This function tests the call function of the separation matrix object
    # by asserting the addition theorem (3.78)in Multiple scattering Interaction of time-harmonic waves with N obstacles
    # by P. A. Martin
    # let b be a separation vector
    # We test the relation
    # sum_{p,t} S_{m,p}^{n,t} * conj( S_{p,N}^{t,M}) = delta_{mM} * delta_{nN}
    b = np.array([3.2, 0.4, -1.35])
    f = 1.0
    rad = 1.0

    print(" Testing with different parameters")
    n = rd.randint(0, high=5)
    m = rd.randint(-n, high=n + 1)

    N = rd.randint(0, high=5)
    M = rd.randint(-N, high=N + 1)
    print(n, m, N, M)

    mat = material.material(1.0)
    part = particle.particle(np.array([0.0, 0.0, 0.0]), rad, mat)
    val = 0.0 + 0j
    sepmat1 = sep.separation_matrix(1.0, 0, 0, 0, 0)
    sepmat2 = sep.separation_matrix(1.0, 0, 0, 0, 0)

    for p in range(11):
        for t in np.arange(-p, p + 1):
            sepmat1.reset_sepmat(1.0, n, m, p, t)
            sepmat2.reset_sepmat(1.0, p, t, N, M)

            spval1 = sepmat1(b, f, part, type='out')
            spval2 = sepmat1(b, f, part, type='out')

            val = val + spval1 * np.conj(spval2)
            print(val)

    assert val == float(M == m and N == n)

    print(" Now testing with identical parameters")
    n = rd.randint(0, high=5)
    m = rd.randint(-n, high=n + 1)
    N = n
    M = m
    print(n, m, N, M)

    mat = material.material(1.0)
    part = particle.particle(np.array([0.0, 0.0, 0.0]), rad, mat)
    val = 0.0 + 0j
    sepmat1 = sep.separation_matrix(1.0, 0, 0, 0, 0)
    sepmat2 = sep.separation_matrix(1.0, 0, 0, 0, 0)

    for p in range(11):
        for t in np.arange(-p, p + 1):
            sepmat1.reset_sepmat(1.0, n, m, p, t)
            sepmat2.reset_sepmat(1.0, p, t, N, M)

            spval1 = sepmat1(b, f, part, type='out')
            spval2 = sepmat1(b, f, part, type='out')

            val = val + spval1 * np.conj(spval2)
            print(val)

    assert val == float(M == m and N == n)


def test_addition_theorem():
    # This function tests the call function of the separation matrix object
    # by asserting the addition theorem (3.78)in Multiple scattering Interaction of time-harmonic waves with N obstacles
    # by P. A. Martin (z-lib.org)
    # Let r_2 = r_1 + b
    # j_n(kr_2) Y_n,m(r_2) = sum_{s,t} S_{ns}^{m,t} (b) j_s(kr_1) Y_s,t(r_1)

    r1 = np.array([2.34, -1.34, -4.23])
    r2 = np.array([-1.45, 0.134, -0.765])
    b = r2 - r1

    f = 1.0
    rad = 1.0

    n = 0
    m = 0

    mat = material.material(1.0)
    part = particle.particle(np.array([0.0, 0.0, 0.0]), rad, mat)
    wf1 = swf.sph_wf_symbol(1, n, m)
    norm1 = wf1.norm(f, part, functype='background')
    ref = norm1 * wf1(r1, f, part)[1][0]
    val = 0.0 + 0j
    print(ref)
    sepmat = sep.separation_matrix(1.0, 0, 0, 0, 0)
    for p in range(11):
        for t in np.arange(-p, p + 1):
            sepmat.reset_sepmat(1.0, n, m, p, t)
            spval = sepmat(b, f, part, type='out')
            if spval != 0:
                wf2 = swf.sph_wf_symbol(f, p, t)
                wfval = wf2(r2, f, part)[1]
                norm2 = wf2.norm(f, part, functype='background')
                val = val + norm2 * spval * wfval[0]
                print(val)

    assert val == pytest.approx(ref)
