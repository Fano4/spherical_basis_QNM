# distutils: language = c++
# distutils: sources = C_MathFunctions/angular_int_aux.cpp

cimport numpy as np
import scipy.special as sp
PI = 3.14159265358979

cdef extern from "C_MathFunctions/mathfunctions.h":
    double rYlm(int l, int m, double thet, double phi)
    void cart_to_spher(double * x, double * y, double * z, double * r, double * t, double *f, int length)
    void square_root(double *Rez,double *Imz,int len)
    double three_Ylm_integ(int l1, int l2, int l3, int m1, int m2, int m3)

def psquare_root(np.ndarray[double, ndim=1,mode="c"] Rez, np.ndarray[double, ndim=1,mode="c"] Imz):
    square_root(&Rez[0],&Imz[0],len(Rez))
    print(Rez,Imz)
    return Rez+1j*Imz

def psph_Bessel_ovlp(l: int, k: complex, kp: complex,r: double) -> complex:

    if k == kp:
        return ( r**2 * PI ) * ( sp.jv(0.5+l,r * k)**2 - sp.jv(-0.5+l,r * k) * sp.jv(1.5+l,r * k)) / ( 4 * k)
    else:
        return ( r**2 * PI ) * ( k * sp.jv(-0.5+l,r * k) * sp.jv(0.5+l,r * kp) - kp * sp.jv(-0.5+l,r * kp) * sp.jv(0.5+l,r * k)) / ( 2 * (k * kp)**0.5 * r * (kp**2 - k**2))

def pYlm(l, m, thet, phi):
    if m < 0:
        return (0.5)**0.5 * ( rYlm(l,abs(m),thet,phi) - 1j * rYlm(l,-abs(m),thet,phi))
    elif m == 0:
        return rYlm(l, 0, thet, phi)
    else:
        return (-1)**m * (0.5) ** 0.5 * (rYlm(l, abs(m), thet, phi) + 1j * rYlm(l, -abs(m), thet, phi))

def pspherical_wave_function(l: int, m: int, z, thet, phi, trans: bool = False) -> complex:
    if trans:
        return sp.spherical_jn(l, z) * (-1)**m * pYlm(l, -m, thet, phi)
    else:
        return sp.spherical_jn(l,z) * pYlm(l,m,thet,phi)

def poutgo_spherical_wave_function(l: int, m: int, z: complex, thet: double, phi: double, trans: bool = False)-> complex:
    if trans:
        return (sp.spherical_jn(l, z) + 1j * sp.spherical_yn(l, z)) * (-1)**m * pYlm(l, -m, thet, phi)
    else:
        return (sp.spherical_jn(l,z) + 1j * sp.spherical_yn(l,z)) * pYlm(l,m,thet,phi)

def pcart_to_spher(np.ndarray[double, ndim=1,mode="c"] x,np.ndarray[double, ndim=1,mode="c"] y,np.ndarray[double, ndim=1,mode="c"] z,np.ndarray[double, ndim=1,mode="c"] r,np.ndarray[double, ndim=1,mode="c"] t,np.ndarray[double, ndim=1,mode="c"] f):
    cart_to_spher(&x[0],&y[0],&z[0],&r[0],&t[0],&f[0],len(x))
    pass

def pgaunt_coeff(p,t,n,m,q):
    return (-1) ** (t+m) * three_Ylm_integ( p, n, q, t, m, -t-m )

