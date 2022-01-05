# distutils: language = c++
# distutils: sources = C_MathFunctions/angular_int_aux.cpp

cimport numpy as np
import scipy.special as sp
PI = 3.14159265358979

cdef extern from "C_MathFunctions/mathfunctions.h":
    void cart_to_spher(double * x, double * y, double * z, double * r, double * t, double *f, int length)
    void square_root(double *rez, double *imz, int length)

def psquare_root(np.ndarray[double, ndim=1, mode="c"] rez, np.ndarray[double, ndim=1, mode="c"] imz):
    square_root(&rez[0], &imz[0], len(rez))
    return rez + 1j * imz

def psph_bessel_ovlp(l: int, k: complex, kp: complex, r: double) -> complex:
    if k == kp:
        return (r ** 2 * PI) * (sp.jv(0.5 + l, r * k) ** 2 - sp.jv(-0.5 + l, r * k) * sp.jv(1.5 + l, r * k)) / (4 * k)
    else:
        return (r ** 2 * PI) * (
                    k * sp.jv(-0.5 + l, r * k) * sp.jv(0.5 + l, r * kp) - kp * sp.jv(-0.5 + l, r * kp) * sp.jv(0.5 + l,
                                                                                                               r * k)) / (
                           2 * (k * kp) ** 0.5 * r * (kp ** 2 - k ** 2))

def pspherical_wave_function(l: int, m: int, z, thet, phi, trans: bool = False) -> complex:
    if trans:
        return sp.spherical_jn(l, z) * (-1) ** m * sp.sph_harm(-m, l, thet, phi)
    else:
        return sp.spherical_jn(l, z) * sp.sph_harm(m, l, thet, phi)

def poutgo_spherical_wave_function(l: int, m: int, z: complex, thet: double, phi: double, trans: bool = False)-> complex:
    if trans:
        return (sp.spherical_jn(l, z) + 1j * sp.spherical_yn(l, z)) * (-1) ** m * sp.sph_harm(-m, l, thet, phi)
    else:
        return (sp.spherical_jn(l, z) + 1j * sp.spherical_yn(l, z)) * sp.sph_harm(m, l, thet, phi)

def pcart_to_spher(np.ndarray[double, ndim=1,mode="c"] x,np.ndarray[double, ndim=1,mode="c"] y,np.ndarray[double, ndim=1,mode="c"] z,np.ndarray[double, ndim=1,mode="c"] r,np.ndarray[double, ndim=1,mode="c"] t,np.ndarray[double, ndim=1,mode="c"] f):
    cart_to_spher(&x[0],&y[0],&z[0],&r[0],&t[0],&f[0],len(x))
    pass
