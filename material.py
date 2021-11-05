import numpy as np
import mathfunctions

# The equations and constants used have been published in https://doi.org/10.1117/1.JNP.8.083097

um_scale = 1  # um_scale = 1 sets the reference length to 1 micrometer
eV_um_scale = um_scale / 1.23984193
hz_um_scale = um_scale / 2.9979246e14


def eps_drude(fp, gam, f):
    # TODO: Unit testing eps drude
    f = f / hz_um_scale
    return - fp ** 2 / (f ** 2 + 1j * gam * f)


def eps_lorentz(deps, fl, gaml, f):
    # TODO: Unit testing eps Lorentz
    f = f / hz_um_scale
    return - deps * fl ** 2 / (f ** 2 - fl ** 2 + 1j * gaml * f)


class material:
    # TODO unit testing class material
    def __init__(self, mat):
        self.drude_param = {}
        self.lorentz_param = {}
        self.epsval = 1

        if isinstance(mat, (int, float)):
            self.epsval = mat

        elif mat == 'Au':
            # TODO wavelength span 400-800nm
            self.epsinf = 6.15991
            self.drude_param = {
                'wp': 1.3457e16,
                'gam': 1.66938e15
            }
            self.lorentz_param = {
                'deps': 2.07122,
                'wp': 4.66171e15,
                'gam': 7.20958e13
            }
        else:
            print("Error: Unknown material")
        pass

    def eps(self, f):
        # TODO unit testing
        if len(self.drude_param) != 0:
            w = f
            return self.epsinf + eps_drude(self.drude_param['wp'], self.drude_param['gam'], f) \
                   + eps_lorentz(self.lorentz_param['deps'], self.lorentz_param['wp'], self.lorentz_param['gam'], f)
        else:
            return self.epsval

    def n(self, f):
        # TODO: Unit testing refraction index
        Rez = np.real(self.eps(f))
        Imz = np.imag(self.eps(f))
        if not isinstance(Rez, np.ndarray):
            Rez = np.array([Rez])
            Imz = np.array([Imz])
        mathfunctions.psquare_root(Rez, Imz)
        return Rez + 1j * Imz

    def k(self, f):
        # TODO: Unit testing wavenumber
        return self.n(f) * 2 * np.pi * f
