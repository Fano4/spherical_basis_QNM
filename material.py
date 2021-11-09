import numpy as np
import mathfunctions

# The equations and constants used have been published in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4875142/pdf/11468_2015_Article_128.pdf

eps0 = 8.85378e-12
um_scale = 1  # um_scale = 1 sets the reference length to 1 micrometer
eV_um_scale = 1.23984193 / um_scale
hz_um_scale = 2.9979246e14 / um_scale


def eps_drude(fp, gam, f):
    f = f * hz_um_scale
    eps = - fp ** 2 / (f ** 2 + 1j * gam * f)
    return eps


# def eps_lorentz(deps, fl, gaml, f):
#    f = f * hz_um_scale
#    return - deps * fl ** 2 / (f ** 2 - fl ** 2 + 1j * gaml * f)


class material:
    def __init__(self, mat):
        self.drude_param = {}
        self.lorentz_param = {}
        self.epsval = 1

        if isinstance(mat, (int, float)):
            self.epsval = mat

        elif mat == 'Au':
            # TODO wavelength span 400-800nm
            self.epsinf = 1
            self.drude_param = {
                'wp': 2.187843711327759e15,
                'gam': 1.7483323775316169e13
            }
        #            self.lorentz_param = {
        #                'deps': 2.07122,
        #                'wp': 4.66171e15,
        #                'gam': 7.20958e13
        #            }
        else:
            print("Error: Unknown material")
        pass

    def eps(self, f):
        # TODO unit testing (Against data ? Find the data)
        if len(self.drude_param) != 0:
            return self.epsinf + eps_drude(self.drude_param['wp'], self.drude_param['gam'], f)
        #                   + eps_lorentz(self.lorentz_param['deps'], self.lorentz_param['wp'], self.lorentz_param['gam'], f)
        else:
            return self.epsval

    def n(self, f):
        # TODO: Unit testing refraction index (Against data? find the data )
        Rez = np.real(self.eps(f))
        Imz = np.imag(self.eps(f))

        if isinstance(f, np.ndarray):
            Rez = np.array([Rez])
            Imz = np.array([Imz])
            mathfunctions.psquare_root(Rez, Imz)
            return Rez + 1j * Imz
        else:
            return float(Rez) + 1j * float(Imz)

    def k(self, f):
        n = self.n(f)
        k = n * 2 * np.pi * f
        return k
