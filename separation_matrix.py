import spherical_wave_function as swf
import mathfunctions
import particle
import numpy as np


class separation_matrix:
    def __init__(self, p, n, t, m):
        self.p = p
        self.n = n
        self.t = t
        self.m = m
        self.q0 = 0
        self.bigq = 0
        self._set_q0()
        self._set_bigq()

        self.sph_wf = []
        self.gaunt_symbol_vec = np.zeros((self.bigq+1, 5))
        for q in range(self.bigq+1):
            self.sph_wf.append(swf.sph_wf_symbol((-1) ** q, self.q0 + 2 * q, t - m))
            self.gaunt_symbol_vec[q] = np.array([p, t, n, -m, self.q0 + 2 * q])

    def _set_q0(self):
        p = self.p
        n = self.n
        t = self.t
        m = self.m

        if np.abs(p - n) >= np.abs(t + m):
            self.q0 = np.abs(p - n)
        elif (p + n + np.abs(t + m)) % 2 == 0:
            self.q0 = np.abs(p + m)
        else:
            self.q0 = np.abs(p + m) + 1
        return

    def _set_bigq(self):
        p = self.p
        n = self.n
        q0 = self.q0
        self.bigq = (p + n - q0) / 2
        return

    def __call__(self, **kwargs):

        # parameter type controls the type of spherical basis function used in the expansion.
        # type=0 => hankel ; type=1 => bessel

        origin = np.array((0,0,0))
        val = np.zeros(self.bigq+1,dtype=complex)
        prefactor = 4 * np.pi * (-1)**(self.n + self.m + self.bigq)
        try:
            for q in range(self.bigq+1):
                if not kwargs['type']:
                    val[q] = self.sph_wf[q].outgo_form(kwargs['r'], kwargs['f'], kwargs['medium'],
                                                       particle.particle(origin, 0, kwargs['medium']))
                else:
                    val[q] = self.sph_wf[q](kwargs['r'], kwargs['f'], kwargs['particle'])[1]
                arg = self.gaunt_symbol_vec[q]
                val[q] = val[q] * mathfunctions.pgaunt_coeff(arg[0], arg[1], arg[2], arg[3], arg[4])

        except KeyError:
            print("Missing keyword argument.")
            print(kwargs.keys())

        val = val * prefactor

        return val
