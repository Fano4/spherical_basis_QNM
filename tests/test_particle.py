import numpy as np
from numpy import random as rd

from src.spherical_basis_QNM.material import material
from src.spherical_basis_QNM.particle import particle


def test_cart_sph_transform():
    pos = np.array([1.4, 3.2, 2.6])
    R = 2
    mat = material.material('Au')
    p = particle.particle(pos, R, mat)

    r = np.array([1.5, np.pi / 3, 3 * np.pi / 5])
    x = np.array([r[0] * np.sin(r[1]) * np.cos(r[2]), r[0] * np.sin(r[1]) * np.sin(r[2]), r[0] * np.cos(r[1])])
    x = x + pos

    sph = p.cart_sph_cen_coord(x)

    assert (sph[0] - r[0] < 1e-10 and sph[1] - r[1] < 1e-10 and sph[2] - r[2] < 1e-10)


def test_inout():
    R = 10
    cen = np.array([-1.3, 2.4, 6.5])
    p = particle.particle(cen, R, material.material('Au'))

    d = rd.rand(3, 10)
    r = np.array([R * d[0], np.pi * d[1], 2 * np.pi * d[2]])
    points_in = np.array([r[0] * np.sin(r[1]) * np.cos(r[2]) + cen[0], r[0] * np.sin(r[1]) * np.sin(r[2]) + cen[1],
                          r[0] * np.cos(r[1]) + cen[2]]).T
    r = np.array([10 + 5 * R * d[0], np.pi * d[1], 2 * np.pi * d[2]])
    points_out = np.array([r[0] * np.sin(r[1]) * np.cos(r[2]) + cen[0], r[0] * np.sin(r[1]) * np.sin(r[2]) + cen[1],
                           r[0] * np.cos(r[1]) + cen[1]]).T

    assert np.all(p.inout(points_in))
    assert not np.any(p.inout(points_out))
