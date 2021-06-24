import numpy as np
import mathfunctions

um_scale = 0.1
eV_um_scale = um_scale/1.23984193

class material:
    def __init__(self,mat):

        if mat == 'Au':
            self.wp = 8.926904839370055 # eV Novotny page 380
            self.gam = 0.07045803673417018 #eV Novotny page 380
        else:
            print("Error: Unknown material")
        pass

    def eps(self,f):
        w = f
        wp = eV_um_scale * self.wp
        gam = eV_um_scale * self.gam
        return 1 - wp**2 / ( w**2 +1j * gam * w)

    def n(self,f):
        Rez = np.real(self.eps(f))
        Imz = np.imag(self.eps(f))
        mathfunctions.psquare_root(Rez, Imz)
        return Rez + 1j*Imz

    def k(self,f):
        return self.n(f) * 2 * np.pi * f
