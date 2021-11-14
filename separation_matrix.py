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
import warnings
'''
def list_app(func):
    def wrapper(*args):
        save=args[0]
        print(args)
        isarray = False
        size=1

        for i in args[1:]:
            if isinstance(i,np.ndarray):
                size = len(i)
                isarray = True
        if isarray:
            newargs = []
            for i in args[1:]:
                if isinstance(i, np.ndarray):
                    newargs.append(i)
                else:
                    newargs.append([i]*size)
            args = np.array(newargs).T
            res = []
            for i in range(args.shape[0]):
                arglist = args[i].tolist()
                res.append(func(save, *arglist))
            print(type(res[0]))
            return res
        else:
            return func(*args)
    return wrapper
'''


class separation_matrix(swf.sph_wf_symbol):

    def __init__(self, a0, p, t, n, m):
        # TODO Test constructor in separation matrix class

        if (n < 0 or abs(m) > n) or (p < 0 or abs(t) > p):
            a0 = 0
            n = 0
            m = 0
            p = 0
            t = 0
        if isinstance(n, np.ndarray) and len(n) == 1:
            n = n[0]
            m = m[0]

        q0 = self.set_q0(p, t, n, m)
        bigq = self.set_bigq(p, n, q0)

        a = np.zeros(bigq + 1, dtype=complex)
        l = np.zeros(bigq + 1, dtype=int)
        ml = np.zeros(bigq + 1, dtype=int)

        prefactor = 4 * np.pi * (-1) ** (n + m + bigq)

        if bigq + 1 < 0:
            raise ValueError("Invalid bigq value!!! Find out how this happened")

        for q in range(int(bigq + 1)):
            try:
                gaunt_symbol_vec = np.array([p, t, n, -m, q0 + 2 * q])
                a[q] = a0 * prefactor * (-1) ** q * mathfunctions.pgaunt_coeff(gaunt_symbol_vec[0], gaunt_symbol_vec[1],
                                                                               gaunt_symbol_vec[2],
                                                                               gaunt_symbol_vec[3], gaunt_symbol_vec[4])
            except:
                raise RuntimeError("Error during Gaunt symbol evaluation")
            l[q] = q0 + 2 * q
            ml[q] = t - m

        super().__init__(a, l, ml)

        return

    def __call__(self, **kwargs):
        # TODO: Test call in separation matrix class
        # TODO: Fix the type parameter by being more consistent with the parent class
        # parameter type controls the type of spherical basis function used in the expansion.
        # type=0 => hankel ; type=1 => bessel

        origin = np.array((0, 0, 0))
        try:

            if not kwargs['type']:
                val = self.outgo_form(kwargs['r'], kwargs['f'], kwargs['medium'],
                                      particle.particle(origin, 0, kwargs['medium']))
                return np.sum(val[0] * val[1])
            else:
                val = super().__call__(kwargs['r'], kwargs['f'], kwargs['particle'])
                return np.sum(val[1])

        except KeyError:
            print("Missing keyword argument.")
            print(kwargs.keys())
            raise KeyError()

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
                        val[i] = val[i] + self.a[j] / super().norm(basis_set.part.k(f), basis_set.part)[0]
                    else:
                        val[i] = val[i] + self.a[j] / super().norm(basis_set.part[j].k(f), basis_set[j].part)[0]
        return val

    def set_q0(self, p, t, n, m):
        # TODO: Test set_q0 in separation matrix class

        if np.abs(p - n) >= np.abs(t + m):
            q0 = int(np.abs(p - n))
        elif (p + n + np.abs(t + m)) % 2 == 0:
            q0 = int(np.abs(p + m))
        else:
            q0 = int(np.abs(p + m) + 1)
        return q0

    def set_bigq(self, p, n, q0):
        # TODO: Test set_bigq in separation matrix class
        bigq = int((p + n - q0) / 2)
        return bigq
