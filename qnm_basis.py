import numpy as np
from scipy import integrate as integ
from scipy import linalg as lg
import green_function
from tqdm import tqdm

# quantize the quasi-normal modes as described by Franke et al. Phys rev res 2020
import plot_functions


class qnm_basis:
    def __init__(self, basis_set, coeff, energs):
        self.basis = basis_set
        self.energs = energs
        self.size = len(energs)
        self.coeff = coeff.reshape(self.size, self.basis.size_ini, 3)
        self.bg_green = green_function.green_function(self.basis)
        self.qnm_ovlp = self.qnm_modes_overlap()

        self.chi = self.qnm_chi_mat()
        self.chi_p = np.real(self.chi)
        self.chi_m = np.imag(self.chi)

        self.dipole_array = self.dipole()

        return

    def basis_inner_overlap(self, f):
        basis_ovlp = np.zeros((self.basis.size_ini, self.basis.size_ini), dtype=complex)
        eps_i = np.imag(self.basis.part.mat.eps(f))

        for i in range(self.basis.size_ini):
            for j in range(self.basis.size_ini):
                basis_ovlp[i, j] = self.basis.overlap_matrix(self.energs[i], self.energs[j], eps_i)[i, j]

        ovlp_in = np.tensordot(np.conj(self.coeff), np.tensordot(basis_ovlp, self.coeff, axes=[1, 1])
                               , axes=[[1, 2], [0, 2]])
        return ovlp_in

    def basis_outer_overlap(self, f):
        im_gf = f / (6 * np.pi)
        deps = self.basis.part.mat.eps(f) - self.basis.part.med.eps(f)
        basis_int_mat = f ** 2 * deps * self.basis.basis_fun_integral(f)
        unnorm_qnm_dip = np.tile(1 / self.energs ** 0.5, [3, 1]).T * np.tensordot(self.coeff, basis_int_mat,
                                                                                  axes=[1, 0])
        ovlp = f * im_gf * np.tensordot(unnorm_qnm_dip, np.conj(unnorm_qnm_dip), axes=[1, 1])
        return ovlp

    def qnm_spatial_overlap(self, f, row, col):
        res = np.zeros((self.size, self.size), dtype=complex)
        spatial_fac = self.basis_inner_overlap(f) + self.basis_outer_overlap(f)

        for i in range(self.size):
            for j in range(self.size):
                ai = f / (2 * (self.energs[i] - f))
                aj = f / (2 * (self.energs[j] - f))

                res[i, j] = ai * np.conj(aj) / (np.pi * (np.real(self.energs[i]) * np.real(self.energs[j])) ** 0.5)
        return (res * spatial_fac)[row, col]

    def qnm_modes_overlap(self):
        arr = np.zeros((self.size, self.size), dtype=complex)
        for row in range(self.size):
            for col in tqdm(range(self.size)):
                plot_functions.plot_func(lambda f: self.qnm_spatial_overlap(f, row, col), np.linspace(1, 10))
                val = integ.quad(lambda f: np.real(self.qnm_spatial_overlap(f, row, col)), 1, 10)
                arr[row, col] = val[0]
                print(val[1])
                val = integ.quad(lambda f: np.imag(self.qnm_spatial_overlap(f, row, col)), 1, 10)
                arr[row, col] = arr[row, col] + 1j * val[0]
                print(val[1])
        return arr

    def qnm_chi_mat(self):
        qnm_s_sqrt = lg.sqrtm(self.qnm_ovlp)
        inv_qnm_s_sqrt = lg.inv(qnm_s_sqrt)
        chi = np.matmul(inv_qnm_s_sqrt, np.matmul(np.diag(self.energs), qnm_s_sqrt))
        return chi

    def dipole(self):
        deps = np.zeros(self.size, dtype=complex)
        basis_integr = np.zeros((self.size, self.basis.size_ini), dtype=complex)
        qnm_s_sqrt = lg.sqrtm(self.qnm_ovlp)
        qnm_int = np.zeros((self.size, 3), dtype=complex)
        renorm = np.outer(np.real(self.energs), 1 / np.real(self.energs)) ** 0.5

        for i in range(self.size):
            deps[i] = self.basis.part.mat.eps(self.energs[i]) - self.basis.part.med.eps(self.energs[i])
            basis_integr[i] = self.energs[i] ** 1.5 * self.basis.basis_fun_integral(self.energs[i])
            qnm_int[i] = deps[i] * np.tensordot(basis_integr[i], self.coeff[i], axes=[0, 0])

        deps = np.tile(deps, (3, 1)).T
        qnm_sym = np.tensordot(renorm * qnm_s_sqrt, qnm_int, axes=[1, 0])
        basis_dipole = qnm_sym * deps
        return basis_dipole
