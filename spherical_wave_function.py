import numpy as np
import material
import particle
import mathfunctions


class sph_wf_symbol:
    """This class implements a symbolic representation of linear combinations of spherical wave functions
    The linear combination is of the form sum_i a_i \\psi^{l_i,m_i}
    the values of a_i,l_i and m_i are stored in distinct numpy arrays."""

    def __init__(self, a=np.array([1]), l=np.array([0]), m=np.array([0])):
        if not isinstance(l, np.ndarray):
            self.a = np.array([a])
            self.l = np.array([l])
            self.m = np.array([m])
        else:
            self.a = a
            self.l = l
            self.m = m
        self.length = len(self.a)
        self._check_values()

    def __call__(self, r: float, f: complex, part: particle):
        S = part.inout(r)
        k = part.k(f)
        values = np.zeros(self.length, dtype=complex)
        norm_cst = np.zeros(self.length, dtype=complex)

        if self.length == 0:
            return 0

        for i in range(self.length):
            a = self.a[i]
            l = self.l[i]
            m = self.m[i]
            sph = part.cart_sph_cen_coord(r)
            modsqr = mathfunctions.psph_Bessel_ovlp(l, k, k, part.R)
            norm_cst[i] = 1 / mathfunctions.psquare_root(np.real(modsqr), np.imag(modsqr))
            values[i] = S * mathfunctions.pspherical_wave_function(l, m, k * sph[0], sph[1], sph[2])
        self._check_values()
        return np.array([norm_cst, values])

    def __add__(self, other):
        a = np.concatenate([self.a, other.a])
        l = np.concatenate([self.l, other.l])
        m = np.concatenate([self.m, other.m])
        return sph_wf_symbol(a, l, m)

    __radd__ = __add__

    def __sub__(self, other):
        a = np.concatenate([self.a, -other.a])
        l = np.concatenate([self.l, other.l])
        m = np.concatenate([self.m, other.m])
        return sph_wf_symbol(a, l, m)

    def __neg__(self):
        return sph_wf_symbol(-self.a, self.l, self.m)

    def __mul__(self, other: (float, complex, int)):
        # if isinstance(other, (float,complex,int) ):
        a = other * self.a
        l = self.l
        m = self.m
        # else:
        #    raise TypeError("Undefined spherical_wavefunction multiplication occured")
        self._check_values()
        return sph_wf_symbol(a, l, m)

    __rmul__ = __mul__

    def outgo_form(self, r: float, f: complex, medium: material, part: particle):
        k = medium.k(f)
        values = np.zeros(self.length, dtype=complex)
        norm_cst = np.ones(self.length, dtype=complex)
        for i in range(self.length):
            a = self.a[i]
            l = self.l[i]
            m = self.m[i]
            sph = part.cart_sph_cen_coord(r)  # Can be a fictive particle that serve as origin of the frame
            values[i] = mathfunctions.pspherical_wave_function(l, m, k * sph[0], sph[1], sph[2])
        return np.array([norm_cst, values])

    def _check_values(self):

        if not (self.a.shape == self.l.shape and self.a.shape == self.m.shape):
            print("Error: Linear combination of spherical basis function defined with arrays of different lengths")
            print(self.a.shape, self.l.shape, self.m.shape)
            exit()
        for i in range(self.length):
            if self.l[i] < 0:
                self.a[i] = 0
                self.l[i] = 0
                self.m[i] = 0
            elif np.abs(self.m[i]) > self.l[i]:
                self.a[i] = 0
                self.l[i] = 0
                self.m[i] = 0

        self.length = len(self.a)
        pass

    # TODO Implement a function that reduces the representation by explicitely adding the sph_wf with same l and m

    def sph_deriv(self, sph_comp: int):
        """sph_comp = [-1,0,1]
        This routine returns the derivative array of a spherical basis function in the form of
        an array with three columns : coefficient, l and m for the new functions
        -1 : ddm1 = -1/k(ddx - i ddy)
        0 : dd0 = -1/k (ddz)
        +1 : ddp1 = -1/k(ddx + i ddy)"""

        l = self.l
        m = self.m

        if sph_comp != 0:
            ss = np.sign(sph_comp)
            self.a = np.concatenate([- ss * ((l + ss * m + 2) * (l + ss * m + 1) / ((2 * l + 1) * (2 * l + 3))) ** 0.5,
                                     - ss * ((l - ss * m) * (l - ss * m - 1) / (4 * l ** 2 - 1)) ** 0.5])
            self.l = np.concatenate([l + 1, l - 1])
            self.m = np.concatenate([m + ss, m + ss])

        else:
            self.a = np.concatenate([(((l + 1) ** 2 - m ** 2) / ((2 * l + 1) * (2 * l + 3))) ** 0.5,
                                     - ((l ** 2 - m ** 2) / (4 * l ** 2 - 1)) ** 0.5])
            self.l = np.concatenate([l + 1, l - 1])
            self.m = np.concatenate([m, m])

        self._check_values()
        return self

    def cart_deriv(self, cart_comp: int):
        """This routine returns the derivative vector of a spherical basis function w r t cartesian coordinates
        1/k ddx = -1/2 (ddm1 + ddp1)
        1/k ddy = 1j/2 (ddp1 - ddm1)
        1/k ddz = -dd0 """

        l = self.l
        m = self.m
        dummy = sph_wf_symbol(self.a, self.l, self.m)

        if cart_comp == 0 or cart_comp == 1:
            ddm1 = self.sph_deriv(-1)
            ddp1 = self.sph_deriv(1)
            if cart_comp != 0:
                return 1j * 0.5 * (ddp1 - ddm1)
            else:
                return - 0.5 * (ddp1 + ddm1)

        elif cart_comp == 2:
            dd0 = self.sph_deriv(0)
            return -dd0

        else:
            print(str("Error: Unrecognized cartesian component" + str(cart_comp)))
            exit()

        self._check_values()
        return self


def sph_wf_deriv_tensor(a, l, m):
    return [[sph_wf_symbol(a, l, m).cart_deriv(i).cart_deriv(j) for i in range(3)] for j in range(3)]
