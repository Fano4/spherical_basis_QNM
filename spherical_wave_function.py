import numpy as np
import material
import particle
import mathfunctions
import scipy.special as sp


class sph_wf_symbol:
    """This class implements a symbolic representation of linear combinations of spherical wave functions
    The linear combination is of the form sum_i a_i \\psi^{l_i,m_i}
    the values of a_i,l_i and m_i are stored in distinct numpy arrays."""

    def __init__(self, a=np.array([1]), l=np.array([0]), m=np.array([0])):
        if not isinstance(l, np.ndarray):
            self.a = np.array([a], dtype=complex)
            self.l = np.array([l], dtype=int)
            self.m = np.array([m], dtype=int)
        else:
            self.a = a
            self.l = l
            self.m = m
        self.length = len(self.a)
        self._check_values()

    def __call__(self, r: np.ndarray, f: complex, part: particle, **kwargs):
        k = part.k(f)

        if self.length == 0:
            return 0
        norm_cst = self.norm(k, part)
        values = self.eval(r, f, part, **kwargs)
        self._check_values()
        return [norm_cst, values]

    def norm(self, k, part):
        norm_cst = np.zeros(self.length, dtype=complex)
        for i in range(self.length):
            l = self.l[i]
            modsqr = mathfunctions.psph_Bessel_ovlp(l, k, k, part.R)
            # if isinstance(modsqr, np.ndarray):
            #   norm_cst[i] = 1 / mathfunctions.psquare_root(np.real(modsqr), np.imag(modsqr))  # Branch cut square root
            # elif isinstance(modsqr, complex):
            norm_cst[i] = 1 / np.sqrt(modsqr)  # Branch cut square root
        return norm_cst

    def eval(self, r, f, part, **kwargs):
        k = part.k(f)
        if 'trans' in kwargs and kwargs['trans']:
            trans = True
        else:
            trans = False

        S = part.inout(r)
        values = np.zeros(self.length, dtype=complex)
        if ('inout' not in kwargs) or kwargs['inout'] == 'in':
            for i in range(self.length):
                l = self.l[i]
                m = self.m[i]
                sph = part.cart_sph_cen_coord(r)
                values[i] = S * mathfunctions.pspherical_wave_function(l, m, k * sph[0], sph[1], sph[2], trans)
        elif kwargs['inout'] == 'out':
            for i in range(self.length):
                values[i] = (1. - S) * self.outgo_form(r, f, part.med, part)[1][i]

        return values.tolist()

    def __add__(self, other):
        # TODO: Unit testing overload __add__ in sph_wf_symbol
        a = np.concatenate([self.a, other.a])
        l = np.concatenate([self.l, other.l])
        m = np.concatenate([self.m, other.m])
        self._check_values()
        return sph_wf_symbol(a, l, m)

    __radd__ = __add__

    def __sub__(self, other):
        a = np.concatenate([self.a, -other.a])
        l = np.concatenate([self.l, other.l])
        m = np.concatenate([self.m, other.m])
        self._check_values()
        return sph_wf_symbol(a, l, m)

    def __neg__(self):
        self._check_values()
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

    def outgo_form(self, r: np.ndarray, f: complex, medium: material, part: particle):
        # TODO: Unit testing outgoing form in sph_wf_symbol
        k = medium.k(np.real(f))
        values = np.zeros(self.length, dtype=complex)
        norm_cst = np.ones(self.length, dtype=complex)
        for i in range(self.length):
            l = self.l[i]
            m = self.m[i]
            sph = part.cart_sph_cen_coord(r)  # Can be a fictive particle that serve as origin of the frame
            values[i] = mathfunctions.pspherical_wave_function(l, m, k * sph[0], sph[1], sph[2])
        self._check_values()
        return [norm_cst.tolist(), values.tolist()]

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

        # Add up coefficients for same functions
        for i in range(self.length):
            for j in np.arange(i + 1, self.length):
                if self.l[i] == self.l[j] and self.m[i] == self.m[j]:
                    self.a[i] = self.a[i] + self.a[j]
                    self.a[j] = 0
        # Remove trivial comoponents
        self.l = np.delete(self.l, np.where(self.a == 0))
        self.m = np.delete(self.m, np.where(self.a == 0))
        self.a = np.delete(self.a, np.where(self.a == 0))

        self.length = len(self.a)
        return

    def sph_deriv(self, sph_comp: int):
        # TODO: Unit testing sph_deriv in sph_wf_symbol
        """sph_comp = [-1,0,1]
        This routine returns the derivative array of a spherical basis function in the form of
        an array with three columns : coefficient, l and m for the new functions
        -1 : ddm1 = -1/k(ddx - i ddy)
        0 : dd0 = -1/k (ddz)
        +1 : ddp1 = -1/k(ddx + i ddy)"""

        # print("sph_deriv_function. derivative component = ", sph_comp)
        # self.print_values()
        # print("#####")
        l = self.l
        m = self.m
        if sph_comp != 0:
            ss = np.sign(sph_comp)
            a = np.concatenate([- ss * ((l + ss * m + 2) * (l + ss * m + 1) / ((2 * l + 1) * (2 * l + 3))) ** 0.5,
                                - (ss * ((l - ss * m) * (l - ss * m - 1) / (4 * l ** 2 - 1)) ** 0.5)])
            l = np.concatenate([l + 1, l - 1])
            m = np.concatenate([m + ss, m + ss])

        else:
            a = np.concatenate([(((l + 1) ** 2 - m ** 2) / ((2 * l + 1) * (2 * l + 3))) ** 0.5,
                                - ((l ** 2 - m ** 2) / (4 * l ** 2 - 1)) ** 0.5])
            l = np.concatenate([l + 1, l - 1])
            m = np.concatenate([m, m])

        self._check_values()

        return [a, l, m]

    def print_values(self, ):
        print("spf_wf object with the coefficients : ")
        print("a         l         m")
        print(np.stack([self.a, self.l, self.m]).T)
        return

    def cart_deriv(self, cart_comp: int):
        # TODO: Unit testing cart_deriv in sph_wf_symbol
        """This routine returns the derivative vector of a spherical basis function w r t cartesian coordinates
        1/k ddx = -1/2 (ddm1 + ddp1)
        1/k ddy = 1j/2 (ddp1 - ddm1)
        1/k ddz = -dd0 """
        ddout = 0
        # print("cart_deriv function. derivative component = ", cart_comp)
        # self.print_values()
        # print("#####")
        if cart_comp == 0 or cart_comp == 1:
            ddm1 = sph_wf_symbol(*self.sph_deriv(-1))
            ddp1 = sph_wf_symbol(*self.sph_deriv(1))
            if cart_comp != 0:
                ddout = 1j * 0.5 * (ddp1 - ddm1)
            else:
                ddout = - 0.5 * (ddp1 + ddm1)

        elif cart_comp == 2:
            ddout = -sph_wf_symbol(*self.sph_deriv(0))

        else:
            print(str("Error: Unrecognized cartesian component" + str(cart_comp)))
            exit()

        self._check_values()
        # print("Out values : ")
        # print("a        l        m")
        # print(np.stack((ddout.a, ddout.l, ddout.m)).T)
        return [ddout.a, ddout.l, ddout.m]


def normalization_cst(l, k, R):
    modsqr = mathfunctions.psph_Bessel_ovlp(l, k, k, R)
    if isinstance(modsqr, np.ndarray):
        Rez = np.real(1 / modsqr)
        Imz = np.imag(1 / modsqr)
        mathfunctions.psquare_root(Rez, Imz)
        return Rez + 1j * Imz
    else:
        return 1 / np.sqrt(modsqr)


def sph_wf_deriv_tensor(a, l, m):
    # TODO: Unit testing sph_wf_deriv_tensor in sph_wf_symbol
    tsr = []
    for i in range(3):
        tsr2 = []
        for j in range(3):
            tsr2.append(sph_wf_symbol(*sph_wf_symbol(*sph_wf_symbol(a, l, m).cart_deriv(i)).cart_deriv(j)))
        tsr.append(tsr2)
    return tsr


def med_sph_wf_ovlp(l, part, f):
    # TODO Unit test med_sph_wf_ovlp in sph_wf_symbol
    k = part.k(f)
    kb = part.med.k(f)
    R = part.R
    normk = normalization_cst(l, k, R)
    normkb = normalization_cst(l, kb, R)
    prefac = normk * normkb
    ovlp = prefac * mathfunctions.psph_Bessel_ovlp(l, k, kb, R)

    return ovlp


def space_rad_integ(l, k, kb):
    # TODO Unit test space_rad_integ
    return -1j / kb * 1 / (k ** 2 - kb ** 2) * (k / kb) ** l


def fin_rad_integ(l, part, f):
    # TODO Unit test fin_rad_integ
    R = part.R
    k = part.k(f)
    kb = part.med.k(f)
    normk = normalization_cst(l, k, R)
    normkb = normalization_cst(l, kb, R)
    ovlp_term = med_sph_wf_ovlp(l, part, f) / (normk * normkb)

    # Rez = np.real(1 / (k * kb))
    # Imz = np.imag(1 / (k * kb))
    # mathfunctions.psquare_root(Rez, Imz)

    # prefactor = (1j * np.pi / 2) * (Rez + Imz * 1j) / (k ** 2 - kb ** 2)
    prefactor = (1j * np.pi / 2) * (1 / np.sqrt(k * kb)) / (k ** 2 - kb ** 2)

    num = (k / kb) ** (2 * l + 1)
    if isinstance(num, np.ndarray):
        Rez = np.real(num)
        Imz = np.imag(num)
        mathfunctions.psquare_root(Rez, Imz)
        fac = Rez + Imz * 1j
    else:
        fac = np.sqrt(num)

    main_term = k * R * sp.yv(l + 0.5, kb * R) * sp.jv(l + 1.5, k * R) - kb * R * sp.jv(l + 0.5, k * R) \
                * sp.yv(l + 1.5, kb * R) - fac * 2 / np.pi
    return ovlp_term + prefactor * main_term
