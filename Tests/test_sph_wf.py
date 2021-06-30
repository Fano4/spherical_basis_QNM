import sys
import numpy as np

sys.path.append("/Users/stephan/PycharmProjects/spherical_basis_QNM/")

import spherical_wave_function as swf


def test_init():
    a = np.linspace(0, 2, 10) + 0.3j
    l = np.arange(0, 10)
    m = np.arange(-5, 5)
    c = swf.sph_wf_symbol(a, l, m)

    assert (np.array_equal(c.a, a))
    assert (np.array_equal(c.l, l))
    assert (np.array_equal(c.m, m))

# Spherical wave functions are complicated to test, as they have analytical values that can only be computed using the
# very function we implemented. We will go for sanity checks, testing for the normalization, orthogonality
def test_call_normalization():
    assert(False)
def test_call_orthogonality():
    assert(False)
