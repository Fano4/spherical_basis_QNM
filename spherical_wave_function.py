import numpy as np
import particle
import mathfunctions

if __name__ == '__main__':
    print("Hello World")
    exit()


def spherical_wave_function(l: int, m: int, r: float, f: complex, part: particle):
    S = part.inout(r)
    k = part.k(f)
    modsqr = mathfunctions.psph_Bessel_ovlp(l, k, k, part.R)
    norm = 1 / mathfunctions.psquare_root(np.real(modsqr), np.imag(modsqr))
    sph = part.cart_sph_cen_coord(r)
    fun = mathfunctions.pspherical_wave_function(l, m, sph[0], sph[1], sph[2])
    return S * norm * fun
