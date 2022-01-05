import mpmath
import numpy as np
import scipy.special as sp

from src.spherical_basis_QNM.particle import particle
from src.spherical_basis_QNM.mathfunctions import mathfunctions


class sph_wf_symbol:
    """This class implements a symbolic representation of linear combinations of spherical wave functions
    The linear combination is of the form sum_i a_i \\psi^{l_i,m_i}
    the values of a_i,l_i and m_i are stored in distinct numpy arrays."""

    def __init__(self, a=np.array([1]), l=np.array([0]), m=np.array([0])):
        self.reset(a, l, m)

    # The reset function is also used to initialize the instance. It can also be used to change the parameters
    # of the instance after its initialization
    def reset(self, a=np.array([1]), l=np.array([0]), m=np.array([0])):

        check_val = check_values(a, l, m)
        checka = check_val[0]
        checkl = check_val[1]
        checkm = check_val[2]

        if not isinstance(l, np.ndarray):
            self.a = np.array([checka], dtype=complex)
            self.l = np.array([checkl], dtype=int)
            self.m = np.array([checkm], dtype=int)
        else:
            self.a = checka
            self.l = checkl
            self.m = checkm
        self.length = len(self.a)

    def __call__(self, r: np.ndarray, f: complex, part: particle, **kwargs):

        if 'inout' in kwargs:
            inout = kwargs['inout']
        else:
            inout = 'inout'

        if 'trans' in kwargs and kwargs['trans']:
            trans = True
        else:
            trans = False

        if self.length == 0:
            return list([0.0, 0.0])

        return self.eval(r, f, part, inout, trans)

    def eval(self, r: np.ndarray, f: complex, part: particle, inout, trans):

        norm_cst = self.norm(f, part)
        if part.inout(r) and (inout == 'in' or inout == 'inout'):
            values = self.inner_form(r, f, part, trans)
        elif not part.inout(r) and (inout == 'out' or inout == 'inout'):
            values = self.outgo_form(r, f, part, trans)
        else:
            values = np.zeros(self.length, dtype=complex)

        return [norm_cst.tolist(), values.tolist()]

    def inner_form(self, r: np.ndarray, f: complex, part: particle, trans):
        check_val = check_values(self.a, self.l, self.m)
        self.a = check_val[0]
        self.l = check_val[1]
        self.m = check_val[2]
        k = part.mat.k(f)
        values = np.zeros(self.length, dtype=complex)
        for i in range(self.length):
            l = self.l[i]
            m = self.m[i]
            sph = part.cart_sph_cen_coord(r)
            values[i] = self.a[i] * mathfunctions.pspherical_wave_function(l, m, k * sph[0], sph[1], sph[2], trans)
        return values

    def outgo_form(self, r: np.ndarray, f: complex, part: particle, trans):
        check_val = check_values(self.a, self.l, self.m)
        self.a = check_val[0]
        self.l = check_val[1]
        self.m = check_val[2]
        k = part.med.k(f)
        values = np.zeros(self.length, dtype=complex)
        for i in range(self.length):
            l = self.l[i]
            m = self.m[i]
            sph = part.cart_sph_cen_coord(r)
            values[i] = self.a[i] * mathfunctions.poutgo_spherical_wave_function(l, m, k * sph[0], sph[1], sph[2],
                                                                                 trans)
        return values

    def norm(self, f, part, **kwargs):
        check_val = check_values(self.a, self.l, self.m)
        self.a = check_val[0]
        self.l = check_val[1]
        self.m = check_val[2]
        if 'functype' in kwargs and kwargs['functype'] == 'background':
            k = part.med.k(f)
        else:
            k = part.k(f)
        norm_cst = np.zeros(self.length, dtype=complex)
        for i in range(self.length):
            l = self.l[i]
            modsqr = mathfunctions.psph_bessel_ovlp(l, k, k, part.R)
            norm_cst[i] = 1 / np.sqrt(modsqr)  # Branch cut square root
        return norm_cst

    def __add__(self, other):
        # TODO: Unit testing overload __add__ in sph_wf_symbol
        check_val = check_values(self.a, self.l, self.m)
        self.a = check_val[0]
        self.l = check_val[1]
        self.m = check_val[2]
        a = np.concatenate([self.a, other.a])
        l = np.concatenate([self.l, other.l])
        m = np.concatenate([self.m, other.m])
        return sph_wf_symbol(a, l, m)

    __radd__ = __add__

    def __sub__(self, other):
        check_val = check_values(self.a, self.l, self.m)
        self.a = check_val[0]
        self.l = check_val[1]
        self.m = check_val[2]
        a = np.concatenate([self.a, -other.a])
        l = np.concatenate([self.l, other.l])
        m = np.concatenate([self.m, other.m])
        return sph_wf_symbol(a, l, m)

    def __neg__(self):
        check_val = check_values(self.a, self.l, self.m)
        self.a = check_val[0]
        self.l = check_val[1]
        self.m = check_val[2]
        return sph_wf_symbol(-self.a, self.l, self.m)

    def __mul__(self, other: (float, complex, int)):
        check_val = check_values(self.a, self.l, self.m)
        self.a = check_val[0]
        self.l = check_val[1]
        self.m = check_val[2]
        a = other * self.a
        l = self.l
        m = self.m
        return sph_wf_symbol(a, l, m)

    __rmul__ = __mul__

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
        check_val = check_values(self.a, self.l, self.m)
        a = check_val[0]
        l = check_val[1]
        m = check_val[2]
        if sph_comp != 0:
            ss = np.sign(sph_comp)
            al = np.concatenate([- ss * ((l + ss * m + 2) * (l + ss * m + 1) / ((2 * l + 1) * (2 * l + 3))) ** 0.5,
                                 - (ss * ((l - ss * m) * (l - ss * m - 1) / (4 * l ** 2 - 1)) ** 0.5)])
            ll = np.concatenate([l + 1, l - 1])
            ml = np.concatenate([m + ss, m + ss])

        else:
            al = np.concatenate([(((l + 1) ** 2 - m ** 2) / ((2 * l + 1) * (2 * l + 3))) ** 0.5,
                                 - ((l ** 2 - m ** 2) / (4 * l ** 2 - 1)) ** 0.5])
            ll = np.concatenate([l + 1, l - 1])
            ml = np.concatenate([m, m])

        return check_values(al * a, ll, ml)

    def print_values(self):
        print("spf_wf object with the coefficients : ")
        print("a         l         m")
        print(np.stack([self.a, self.l, self.m]).T)
        return

    def cart_deriv(self, cart_comp: int):
        # TODO: Unit testing cart_deriv in sph_wf_symbol
        check_val = check_values(self.a, self.l, self.m)
        self.a = check_val[0]
        self.l = check_val[1]
        self.m = check_val[2]
        """This routine returns the derivative vector of a spherical basis function w r t cartesian coordinates
        1/k ddx = -1/2 (ddm1 + ddp1)
        1/k ddy = 1j/2 (ddp1 - ddm1)
        1/k ddz = -dd0 """
        ddout = 0
        # print("cart_deriv function. derivative component = ", cart_comp)
        # self.print_values()
        # print("#####")
        if cart_comp == 0 or cart_comp == 1:
            ddm1_param = self.sph_deriv(-1)
            ddm1 = sph_wf_symbol(ddm1_param[0], ddm1_param[1], ddm1_param[2])
            ddp1_param = self.sph_deriv(1)
            ddp1 = sph_wf_symbol(ddp1_param[0], ddp1_param[1], ddp1_param[2])
            if cart_comp != 0:
                ddout = 1j * 0.5 * (ddp1 - ddm1)
            else:
                ddout = - 0.5 * (ddp1 + ddm1)

        elif cart_comp == 2:
            dd0_param = self.sph_deriv(0)
            ddout = -sph_wf_symbol(dd0_param[0], dd0_param[1], dd0_param[2])

        else:
            print(str("Error: Unrecognized cartesian component" + str(cart_comp)))
            exit()

        return [ddout.a, ddout.l, ddout.m]

    def integral(self, k, r):
        check_val = check_values(self.a, self.l, self.m)
        self.a = check_val[0]
        self.l = check_val[1]
        self.m = check_val[2]
        vec = np.zeros(self.length, dtype=complex)
        for i in range(self.length):
            hyperg = mpmath.hyper([(3 + self.l[i]) / 2], [3 / 2 + self.l[i], (5 + self.l[i]) / 2],
                                  -0.25 * k ** 2 * r ** 2)
            float_hyperg = float(mpmath.nstr(hyperg.real, 6)) + 1j * float(mpmath.nstr(hyperg.imag, 6))
            vec[i] = (4 * np.pi) ** 0.5 * 0.5 ** (2 + self.l[i]) * np.pi ** 0.5 * r ** 3 * (k * r) ** self.l[
                i] * sp.gamma(
                (3 + self.l[i]) / 2) \
                     * float_hyperg

        return np.sum(self.a * vec)


def sph_wf_deriv_tensor(a, l, m):
    # TODO: Unit testing sph_wf_deriv_tensor in sph_wf_symbol
    check_val = check_values(a, l, m)
    a = check_val[0]
    l = check_val[1]
    m = check_val[2]
    tsr = []
    for i in range(3):
        tsr2 = []
        for j in range(3):
            tsr2.append(sph_wf_symbol(*sph_wf_symbol(*sph_wf_symbol(a, l, m).cart_deriv(i)).cart_deriv(j)))
        tsr.append(tsr2)
    return tsr


def med_sph_wf_ovlp(sph_wf, part, f) -> np.array:
    """This function implements the finite integral of two spherical bessel functions with
    a k value for the material and for the background. Eq. (7c) in ref 1
    It returns a 1-D numpy array with each element being the result for each l value"""

    normk = sph_wf.norm(f, part, functype='mat')[0]
    normkb = sph_wf.norm(f, part, functype='background')[0]
    prefac = normk * normkb

    k = part.k(f)
    kb = part.med.k(f)
    R = part.R
    ovlp = prefac * mathfunctions.psph_bessel_ovlp(sph_wf.l, k, kb, R)

    return ovlp


def sph_wf_ovlp(sph_wf, part, f1, f2):
    """This function implements the finite integral of two spherical bessel functions in the material
    at two frequencies.
    It returns a 1-D numpy array with each element being the result for each l value"""
    normk = sph_wf.norm(f1, part, functype='mat')
    normkb = sph_wf.norm(f2, part, functype='mat')
    prefac = normk * normkb

    k1 = part.k(f1)
    k2 = part.k(f2)
    R = part.R
    ovlp = prefac * mathfunctions.psph_Bessel_ovlp(sph_wf.l, k1, k2, R)

    return ovlp


def space_rad_integ(l, k, kb):
    # Eq D4 in ref 1
    # TODO: Work on a test for space_rad_int... This one is very hard...
    return -1j / np.real(kb) * 1 / (np.real(k) ** 2 - np.real(kb) ** 2) * (np.real(k) / np.real(kb)) ** l


def fin_rad_integ(sph_wf, part, f):
    # TODO Unit test fin_rad_integ
    # Eq D5 in ref 1
    R = part.R
    k = part.k(f)
    kb = part.med.k(f)
    normk = sph_wf.norm(f, part, functype='mat')
    normkb = sph_wf.norm(f, part, functype='background')
    ovlp_term = med_sph_wf_ovlp(sph_wf, part, f) / (normk * normkb)

    prefactor = (1j * np.pi / 2) * (1 / np.sqrt(k * kb)) / (k ** 2 - kb ** 2)

    l = sph_wf.l
    fac = np.sqrt((k / kb) ** (2 * l + 1))

    main_term = k * R * sp.yv(l + 0.5, kb * R) * sp.jv(l + 1.5, k * R) - kb * R * sp.jv(l + 0.5, k * R) \
                * sp.yv(l + 1.5, kb * R) - fac * 2. / np.pi

    return ovlp_term + prefactor * main_term


def check_values(a, l, m):
    # TODO: sort the values by increasing value of l and m
    if not isinstance(a, np.ndarray):
        a = np.array([a])
        l = np.array([l])
        m = np.array([m])
    elif len(a.shape) == 2 and a.shape[1] == 1:
        a = a.reshape(a.shape[0])
        l = l.reshape(l.shape[0])
        m = m.reshape(m.shape[0])

    if not (a.shape == l.shape and a.shape == m.shape):
        print(a.shape, l.shape, m.shape)
        raise ValueError("Linear combination of spherical basis function defined with arrays of different lengths")
    elif not len(a.shape) == 1:
        print(a.shape, l.shape, m.shape)
        raise ValueError("Spherical basis functions must be defined using 1D array and not ", len(a.shape),
                         "-D arrays")
    else:
        length = len(a)
        for i in range(length):
            if l[i] < 0:
                a[i] = 0
                l[i] = 0
                m[i] = 0
            elif np.abs(m[i]) > l[i]:
                a[i] = 0
                l[i] = 0
                m[i] = 0

        # Add up coefficients for same functions
        for i in range(length):
            for j in np.arange(i + 1, length):
                if l[i] == l[j] and m[i] == m[j]:
                    a[i] = a[i] + a[j]
                    a[j] = 0
        # Remove trivial components
        newl = np.delete(l, np.where(a == 0))
        newm = np.delete(m, np.where(a == 0))
        newa = np.delete(a, np.where(a == 0))

        return [newa, newl, newm]
