import sys, os
import numpy as np

from src.spherical_basis_QNM.material import material

def test_cst_eps():
    eps = 3.2
    c = material.material(eps)
    assert eps == c.eps(0.1)


def test_drude_model():
    wp = 8.926904839370055
    gam = 0.07045803673417018
    w = np.linspace(5e-3, 3,100)
    c = material.material('Au')
    eps = 1 - wp**2 / (w**2 + 1j * gam * w)
    assert(np.sum((eps-c.eps(w)))**2 < 1e-10)

def test_refr_index():
    n = 2.25
    c = material.material(n ** 2)
    assert c.n(0.1) == n
