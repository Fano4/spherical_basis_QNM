# This class implements a decomposition of the background Green's function in a give basis set.
# The class initialization requires  basis set class instance.
#
# Types of basis sets:
#       'spherical' :
#           Based on ref dx.doi.org/10.1364/JOSAB.30.001996 (Ref 1)
#

import numpy as np
from scipy import linalg as lg

import basis_set
import separation_matrix
import spherical_wave_function as swf


class green_function:
    def __init__(self, basis_set: basis_set.basis_set):
        # TODO: Unit test Green's function constructor
        # The green's function depends on the geometry of the system and is decomposed in a bsis set.
        self.basis_set = basis_set
        self.rank = basis_set.size_ini

        self.self_terms = (lg.norm(np.array(basis_set.distance_mat), axis=2) == 0).tolist()
        # The Green's function has two types of terms. The self (single-particle) terms and the scattering terms.

        # The scattering term involves the separation matrix between basis functions on distinct particles and
        # separation matrix between their second derivatives.

        # The self-term involves a principal integral inside the scatterer and an integral outside of the scatterer.

        # The constructor doesn't compute any integral so that the Green's function class is symbolic.

        return

    def __call__(self, *args, **kwargs):
        # TODO: Unit test call Green's function
        f = args[0]

        result = np.zeros((self.rank, self.rank, 3, 3), dtype=complex)

        for bas_i in np.arange(0, self.rank):
            for bas_j in np.arange(0, self.rank):
                if self.self_terms[bas_i][bas_j]:
                    result[bas_i, bas_j] = np.array(self.self_block(bas_i, bas_j, f)).reshape((3, 3))
                else:
                    result[bas_i, bas_j] = np.array(self.scattering_block(bas_i, bas_j, f)).reshape((3, 3))

        return result

    def self_block(self, bas_i, bas_j, f):
        # TODO: Unit test self_block in Green's function
        # The self block method returns a rank-3 matrix with all the components of the dyadic for a couple of
        # basis functions

        if self.basis_set.bas_func == 'spherical':
            val = self_block_spherical(self.basis_set, bas_i, bas_j, f)
            return val
        else:
            raise TypeError("Unsupported basis set type!")

    def scattering_block(self, bas_i, bas_j, f):
        # TODO: Unit test scattering block in Green's function
        if self.basis_set.bas_func == 'spherical':
            return scattering_block_spherical(self.basis_set, bas_i, bas_j, f)
        else:
            raise TypeError("Unsupported basis set type!")


def self_block_spherical(basis_set: basis_set.basis_set, bas_i, bas_j, f, verbose=False):
    # TODO: Unit test self_block_spherical basis in green's function
    # Eqs. (14), (18) and (20) in Ref. 1
    kb = basis_set.part.med.k(f)
    kj = basis_set.part.mat.k(f)
    prefactor = 1j * kb / (4 * np.pi) ** 0.5
    # Principal value A term
    # Diagonal term
    diag_sep = basis_set.basis_separation_mat[bas_i][0].sph_basis_proj(basis_set, f)[bas_j]
    # TODO: Remove legacy code basis_set.sph_wf_norm('mat', f)[bas_i]
    radial_princ_integ_0 = -(1j / kb) * 1 / (kj ** 2 - kb ** 2) * basis_set.sph_wf_norm('mat', f)[bas_i]
    # Dyadic term
    # This part is actually more tricky. It is the separation matrix element with respect to the Hessian.
    # There is one separation matrix per term in each Hessian matrix element.
    # For each component of the Hessian matrix, we seek the coefficients, l and m values. For each sph_wf_symbol
    # of the basis, we get a list of list of np.array

    # Let's project the hessian components on the basis set. Now we get a 2-list containing numpy arrays
    hessian_tens_vectors = [[basis_set.hessian_proj(bas_j, i, j) for i in range(3)] for j in range(3)]

    ll = basis_set.jlm_to_index(bas_i)[1]
    ml = basis_set.jlm_to_index(bas_i)[2]

    # hessian_sep_mat contains the components of the separation matrix for the hessian of the spherical wave function
    # It is a 2-list containing numpy arrays of dimension basis_set.size. The arrays represent the coefficient in the
    # spherical wave function basis

    hessian_sep_mat = [
        [[separation_matrix.separation_matrix(hessian_tens_vectors[i][j][k], ll, ml,
                                              basis_set.basis[k].l - basis_set.jlm_to_index(bas_j)[1],
                                              basis_set.jlm_to_index(bas_j)[2] - basis_set.basis[k].m
                                              ).sph_basis_proj(basis_set, f)[bas_j]
          for k in range(basis_set.size)] for i in range(3)] for j in range(3)]

    radial_princ_integ_l = [[[-(1j / kb) * 1 / (kj ** 2 - kb ** 2) * (kj / kb) ** basis_set.basis[k].l
                              for k in range(basis_set.size)] for i in range(3)] for j in range(3)]

    mval = np.array([basis_set.basis[k].m for k in range(basis_set.size)])
    term_a = [[((i == j) * diag_sep * radial_princ_integ_0
                + (1 / kb ** 2) * np.sum((-1) ** (abs(-mval)) * np.array(hessian_sep_mat[i][j])
                                         * np.array(radial_princ_integ_l[i][j])))
               for i in range(3)] for j in range(3)]

    # Outer B term
    llj = basis_set.jlm_to_index(bas_j)[1]
    mlj = basis_set.jlm_to_index(bas_j)[2]
    jj = basis_set.jlm_to_index(bas_j)[0]
    if isinstance(basis_set.part, list):
        fin_rad_integ = swf.fin_rad_integ(llj, basis_set.part[jj], f)
    else:
        fin_rad_integ = swf.fin_rad_integ(llj, basis_set.part, f)

    prefactor = 1j * kb * basis_set.med_sph_wf_ovlp(f)[bas_i] * fin_rad_integ \
                * basis_set.sph_wf_norm('mat', f)[bas_i] / basis_set.sph_wf_norm('background', f)[bas_j]

    new_hessian_tens_vectors = [[np.zeros(basis_set.size, dtype=complex) for i in range(3)] for j in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(basis_set.size):
                new_l = 2 * llj - basis_set.basis[k].l
                new_m = 2 * mlj - basis_set.basis[k].m
                for kp in range(basis_set.size):
                    if new_l == basis_set.basis[kp].l and new_m == basis_set.basis[kp].m:
                        new_hessian_tens_vectors[i][j][kp] = hessian_tens_vectors[i][j][k]

    ll = basis_set.jlm_to_index(bas_i)[1]
    llp = basis_set.jlm_to_index(bas_j)[1]
    mm = basis_set.jlm_to_index(bas_i)[2]
    mmp = basis_set.jlm_to_index(bas_j)[2]

    temp = [[new_hessian_tens_vectors[i][j][bas_i]
             for i in range(0, 3)] for j in range(0, 3)]
    term_b = [[prefactor * (i == j) * (bas_i == bas_j) + (1 / kb) ** 2 * temp[i][j]
               for i in range(0, 3)] for j in range(0, 3)]

    val = (np.array(term_a) - np.array(term_b)).tolist()
    if verbose:
        return [val, term_a, term_b]
    else:
        return val


def scattering_block_spherical(basis_set: basis_set.basis_set, bas_i, bas_j, f):
    # TODO: Unit test scattering block spheical in green's funciton
    # TODO Check the indices of the separation matrices!
    # Eq. (13) in Ref 1.

    # Pre-factor section
    med_wf_ovlp = basis_set.med_sph_wf_ovlp(f)
    wf_norm = basis_set.sph_wf_norm('background', f)
    kb = basis_set.part.med.k(f)
    prefactor = 1j * kb * med_wf_ovlp[bas_i] * med_wf_ovlp[bas_j] / (wf_norm[bas_i] * wf_norm[bas_j])
    # Diagonal part
    diag_sep = basis_set.basis_separation_mat[bas_i][bas_j](type=0, r=basis_set.distance_mat[bas_i][bas_j],
                                                            f=f, medium=basis_set.part.med)

    # dyadic part
    # This part is actually more tricky. It is the separation matrix element with respect to the Hessian.
    # There is one separation matrix per term in each Hessian matrix element.

    # For each component of the Hessian matrix, we seek the coefficients, l and m values
    # !!! Indices of the separation matrix in Eq. (13)
    amjl = [[basis_set.basis_hessian[bas_j][i][j].a for i in range(0, 3)] for j in range(0, 3)]
    lmjl = [[2 * basis_set.jlm_to_index(bas_j)[1] - basis_set.basis_hessian[bas_j][i][j].l
             for i in range(0, 3)] for j in range(0, 3)]
    mmjl = [[2 * basis_set.jlm_to_index(bas_j)[2] - basis_set.basis_hessian[bas_j][i][j].m
             for i in range(0, 3)] for j in range(0, 3)]

    checked = check_aml_list(amjl, lmjl, mmjl)
    amjl = checked[0]
    lmjl = checked[1]
    mmjl = checked[2]

    # Now, we generate a list of separation matrices for the corresponding components
    hessian_sep_mat = [[separation_matrix.separation_matrix(basis_set.jlm_to_index(bas_i)[1],
                                                            basis_set.jlm_to_index(bas_i)[2], lmjl[i][j], mmjl[i][j])
                        for i in range(0, 3)] for j in range(0, 3)]

    result = []
    for j in range(0, 3):
        result.append(
            [(i == j) * diag_sep + (1 / kb ** 2) *
             np.sum(amjl[i][j] * hessian_sep_mat[i][j](type=0, r=basis_set.distance_mat[bas_i][bas_j],
                                                       f=f, medium=basis_set.part.med))
             for i in range(0, 3)])
    return prefactor * np.array(result)


def check_aml_list(a, l, m):
    for i in range(3):
        for j in range(3):
            for k in range(len(a[i][j])):
                if l[i][j][k] < 0 or abs(m[i][j][k]) > l[i][j][k]:
                    a[i][j][k] = 0
                    l[i][j][k] = 0
                    m[i][j][k] = 0

    return [a, l, m]

