# This class implements a decomposition of the background Green's function in a give basis set.
# The class initialization requires  basis set class instance.
#
# Types of basis sets:
#       'spherical' :
#           Based on ref dx.doi.org/10.1364/JOSAB.30.001996 (Ref 1)
#

import numpy as np
import basis_set
from functools import partial
import spherical_wave_function as swf


class green_function:
    def __init__(self, basis_set: basis_set.basis_set):
        self.basis_set = basis_set
        self.rank = basis_set.size
        for j in range(basis_set.npart):
            for jp in range(basis_set.npart):
                if j == jp:
                    self.self_block(j)
                else:
                    self.scattering_block(j, jp)
                    return
        return

    def self_block(self, part_j):
        if self.basis_set.bas_func == 'spherical':
            return self_block_spherical(self.basis_set, part_j)
        else:
            raise TypeError("Unsupported basis set type!")

    def scattering_block(self, part_j, part_jp):
        if self.basis_set.bas_func == 'spherical':
            return scattering_block_spherical(self.basis_set, part_j, part_jp)
        else:
            raise TypeError("Unsupported basis set type!")


def self_block_spherical(basis_set: basis_set.basis_set, part_j: int):
    # Eqs. (14), (18) and (20) in Ref. 1
    return


def scattering_block_spherical(basis_set: basis_set.basis_set, part_j: int, part_jp: int):
    # Eq. (13) in Ref 1.

    for lj in range(basis_set.n_sph_harm):
        for mj in np.arange(-lj, lj + 1):

            index = basis_set.jlm_to_index(part_j, lj, mj)
            psi = basis_set(part_j, lj, mj)
            med_sph_wf_ovlp_jlm = basis_set.med_sph_wf_ovlp
            norm_wf_b_jlm = partial(basis_set.sph_wf_norm, 'background')

            for ljp in range(basis_set.n_sph_harm):
                for mjp in np.arange(-ljp, ljp + 1):
                    indexp = basis_set.jlm_to_index(part_jp, ljp, mjp)
                    psip = basis_set(part_jp, ljp, mjp)

                    # Each matrix element is a rank-3 tensor. There are 9 cartesian components
                    # that we address individually
                    # xx
                    # xy
                    # xz

                    # yx
                    # yy
                    # yz

                    # zx
                    # zy
                    # zz
