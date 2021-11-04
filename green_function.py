# This class implements a decomposition of the background Green's function in a give basis set.
# The class initialization requires  basis set class instance.
#
# Types of basis sets:
#       'spherical' :
#           Based on ref dx.doi.org/10.1364/JOSAB.30.001996 (Ref 1)
#

import numpy as np
from scipy import linalg as lg
from functools import partial

import basis_set
import material
import particle
import separation_matrix
import spherical_wave_function as swf


class green_function:
    def __init__(self, basis_set: basis_set.basis_set):
        # The green's function depends on the geometry of the system and is decomposed in a bsis set.
        self.basis_set = basis_set
        self.rank = basis_set.size

        self.self_terms = (lg.norm(np.array(basis_set.distance_mat), axis=2) == 0).tolist()
        # The Green's function has two types of terms. The self (single-particle) terms and the scattering terms.

        # The scattering term involves the separation matrix between basis functions on distinct particles and
        # separation matrix between their second derivatives.

        # The self-term involves a principal integral inside the scatterer and an integral outside of the scatterer.

        # The constructor doesn't compute any integral so that the Green's function class is symbolic.

        return

    def __call__(self, *args, **kwargs):

        f = args[0]

        result = np.array((self.rank, self.rank, 3, 3), dtype=complex)

        for bas_i in np.arange(0, self.rank):
            for bas_j in np.arange(0, self.rank):
                if self.self_terms[bas_i][bas_j]:
                    result[bas_i, bas_j] = self.self_block(bas_i, bas_j, f)
                else:
                    result[bas_i, bas_j] = self.scattering_block(bas_i, bas_j, f)

        return result

    def self_block(self, bas_i, bas_j, f):
        # The self block method returns a rank-3 matrix with all the components of the dyadic for a couple of
        # basis functions

        if self.basis_set.bas_func == 'spherical':
            return self_block_spherical(self.basis_set, bas_i, bas_j, f)
        else:
            raise TypeError("Unsupported basis set type!")

    def scattering_block(self, bas_i, bas_j, f):
        if self.basis_set.bas_func == 'spherical':
            return scattering_block_spherical(self.basis_set, bas_i, bas_j, f)
        else:
            raise TypeError("Unsupported basis set type!")


def self_block_spherical(basis_set: basis_set.basis_set, bas_i, bas_j, f):
    # Eqs. (14), (18) and (20) in Ref. 1
    return


def scattering_block_spherical(basis_set: basis_set.basis_set, bas_i, bas_j, f):
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
    amjl = [[basis_set.basis_hessian[bas_j][i][j].a for i in range(0, 3)] for j in range(0, 3)]
    lmjl = [[basis_set.basis_hessian[bas_j][i][j].l for i in range(0, 3)] for j in range(0, 3)]
    mmjl = [[basis_set.basis_hessian[bas_j][i][j].m for i in range(0, 3)] for j in range(0, 3)]

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
    return prefactor * result
