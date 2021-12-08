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
from sympy.physics.wigner import gaunt


class separation_matrix(swf.sph_wf_symbol):
    def __init__(self, a0, p, t, n, m):

        alml = set_alml(a0, p, t, n, m)
        super().__init__(alml[0], alml[1], alml[2])

        return

    def reset_sepmat(self, a0, p, t, n, m):
        alml = set_alml(a0, p, t, n, m)
        super().reset(alml[0], alml[1], alml[2])

        return

    def __call__(self, r: np.ndarray, f: complex, part: particle.particle, **kwargs):

        if 'type' not in kwargs:
            raise ValueError("Evaluating separation matrix requires the type[='in' or 'out] of separation matrix.")

        if 'trans' in kwargs and kwargs['trans']:
            trans = True
        else:
            trans = False

        if kwargs['type'] == 'out':
            val = self.outgo_form(r, f, part, trans)
        elif kwargs['type'] == 'in':
            val = self.inner_form(r, f, part, trans)
        else:
            raise ValueError("Invalid type of separation matrix. Please either use 'in' or 'out'")

        return np.sum(val)

    def sph_basis_proj(self, basis_set, f):
        # TODO: test sph_basis_proj in separation_matrix class
        # Computes the bracket < D | psi > for all the psi in the basis set
        val = np.zeros((basis_set.size), dtype=complex)
        for i in range(basis_set.size):
            lref = basis_set(i).l
            mref = basis_set(i).m

            for j in range(self.length):
                if lref == self.l[j] and mref == self.m[j]:
                    if not isinstance(basis_set.part, list):
                        val[i] = val[i] + self.a[j] / super().norm(basis_set.part.k(f), basis_set.part)[j]
                    else:
                        val[i] = val[i] + self.a[j] / super().norm(basis_set.part[j].k(f), basis_set[j].part)[j]
        return val


def set_q0(p, t, n, m):
    if np.abs(p - n) >= np.abs(t + m):
        q0 = int(np.abs(p - n))
    elif (p + n + np.abs(t + m)) % 2 == 0:
        q0 = int(np.abs(t + m))
    else:
        q0 = int(np.abs(t + m) + 1)
    return q0


def set_bigq(p, n, q0):
    if (p + n - q0) % 2 == 0:
        bigq = int((p + n - q0) / 2)
    else:
        raise ValueError("big_q error. p + n - q0 should be even")
    return bigq


def set_alml(a0, p, t, n, m):
    if isinstance(n, np.ndarray) and len(n.shape) == 1 and len(n) == 1:
        n = n[0]
        m = m[0]
    elif isinstance(n, np.ndarray) and len(n.shape) == 2 and n.shape[1] == 1:
        n = n[0, 0]
        m = m[0, 0]

    if (n < 0 or abs(m) > n) or (p < 0 or abs(t) > p):
        a0 = 0
        n = 0
        m = 0
        p = 0
        t = 0
    q0 = set_q0(p, t, n, -m)
    bigq = set_bigq(p, n, q0)

    a = np.zeros(bigq + 1, dtype=complex)
    l = np.zeros(bigq + 1, dtype=int)
    ml = np.zeros(bigq + 1, dtype=int)

    prefactor = (4 * np.pi) * (-1) ** (n + m + bigq)

    if bigq + 1 < 0:
        raise ValueError("Invalid bigq value!!! Find out how this happened")

    for q in range(int(bigq + 1)):
        try:
            gaunt_symbol_vec = np.array([p, n, q0 + 2 * q, t, -m, -t + m])

            a[q] = a0 * prefactor * (-1) ** q * gaunt(gaunt_symbol_vec[0],
                                                      gaunt_symbol_vec[1],
                                                      gaunt_symbol_vec[2],
                                                      gaunt_symbol_vec[3],
                                                      gaunt_symbol_vec[4],
                                                      gaunt_symbol_vec[5])
        except:
            raise RuntimeError("Error during Gaunt symbol evaluation")
        l[q] = q0 + 2 * q
        ml[q] = t - m
    return [a, l, ml]
