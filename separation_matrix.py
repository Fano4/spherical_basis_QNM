# This class implements a symbolic representation for the separation matrices between spherical wave functions.
# The separation matrices are the expansion coefficients of spherical wave functions centered on one point
# in the basis of spherical wave functions centered at another point.
# The class is implemented according to the equations presented in dx.doi.org/10.1364/JOSAB.30.001996 (Ref 1)
# Eqs. (C1) and (C3). the class uses the mathfunctions library develeoped previously.
#
#   Class initialization and options
#       sep = separation_matrix.separation_matrix(p, n, t, m)
#
#       p, t are the l- and m- values of the final basis function
#       n, m are the l- and m-value of the initial basis function
import spherical_wave_function as swf
import mathfunctions
import particle
import numpy as np


class separation_matrix:
    def __init__(self, p, t, n, m):
        # TODO Test constructor in separation matrix class
        self.p = p
        self.n = n
        self.t = t
        self.m = m
        self.q0 = 0
        self.bigq = 0
        self._set_q0()
        self._set_bigq()
        if (n < 0 or abs(m) > n) or (p < 0 or abs(t) > p):
            self.prefactor = 0
            n = 0
        else:
            self.prefactor = 4 * np.pi * (-1) ** (self.n + self.m + self.bigq)

        self.sph_wf = []
        self.gaunt_symbol_vec = np.zeros((self.bigq + 1, 5))
        for q in range(int(self.bigq + 1)):
            self.sph_wf.append(swf.sph_wf_symbol((-1) ** q, self.q0 + 2 * q, t - m))
            self.gaunt_symbol_vec[q] = np.array([p, t, n, -m, self.q0 + 2 * q])

    def _set_q0(self):
        # TODO: Test set_q0 in separation matrix class
        p = self.p
        n = self.n
        t = self.t
        m = self.m

        if np.abs(p - n) >= np.abs(t + m):
            self.q0 = int(np.abs(p - n))
        elif (p + n + np.abs(t + m)) % 2 == 0:
            self.q0 = int(np.abs(p + m))
        else:
            self.q0 = int(np.abs(p + m) + 1)
        return

    def _set_bigq(self):
        # TODO: Test set_bigq in separation matrix class
        p = self.p
        n = self.n
        q0 = self.q0
        self.bigq = int((p + n - q0) / 2)
        return

    def __call__(self, **kwargs):
        # TODO: Test call in separation matrix class
        # parameter type controls the type of spherical basis function used in the expansion.
        # type=0 => hankel ; type=1 => bessel

        origin = np.array((0,0,0))
        val = np.zeros(self.bigq+1,dtype=complex)

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

        val = np.sum(val) * self.prefactor

        return val

    def sph_basis_proj(self, basis_set, f):
        # TODO: test sph_basis_proj in separation_matrix class
        # Computes the bracket < D | psi > for all the psi in the basis set
        origin = np.array((0, 0, 0))
        val = np.zeros((basis_set.size, self.bigq + 1), dtype=complex)

        for i in range(basis_set.size):
            lref = basis_set(i).l
            mref = basis_set(i).m
            norm_cst = basis_set.sph_wf_norm('mat', f)[i]

            for q in range(self.bigq + 1):
                val[i, q] = norm_cst * (lref == self.sph_wf[q].l) * (mref == self.sph_wf[q].m)
                arg = self.gaunt_symbol_vec[q]
                val[i, q] = val[i, q] * mathfunctions.pgaunt_coeff(arg[0], arg[1], arg[2], arg[3], arg[4])

        val = np.sum(val, axis=1) * self.prefactor

        return val
