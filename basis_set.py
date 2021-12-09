# This class implements a basis set of 3D functions. The basis set can be chosen from a list.
# the parameters required to initialize each basis set depends on the particular basis set.
#
# foo = basis_set.basis_set(type="type"[, ...])
#
# Types of basis set and options:
#       'spherical' :
#           Based on ref dx.doi.org/10.1364/JOSAB.30.001996 (Ref 1)
#           Parameters:
#               'lmax' : positive integer
#               'part' : list of particle class instances
#
#
import numpy as np
import spherical_wave_function as swf
import separation_matrix as sep


class basis_set:
    def __init__(self, **kwargs):
        # TODO: unit testing constructor in basis set
        try:
            self.bas_func = kwargs['type']
            if self.bas_func == 'spherical':
                # Initialize spherical basis functions

                self.sized = False
                self.lmax_ini = kwargs['lmax']
                self.lmax = self.lmax_ini

                if isinstance(kwargs['part'], list):
                    self.npart = len(kwargs['part'])
                    raise ValueError("The method has not been checked for more than one particle")
                else:
                    self.npart = 1

                self.part = kwargs['part']
                self.n_sph_harm_ini = (self.lmax_ini + 1) ** 2
                self.size_ini = self.npart * self.n_sph_harm_ini

                # The Hessian matrix is a rank-3 tensor of the spherical basis function symbols' second derivatives.
                # Each symbol is a component of the Hessian in cartesian coordinates.
                self.basis_hessian = [swf.sph_wf_deriv_tensor(1, self.jlm_to_index(i)[1],
                                                              self.jlm_to_index(i)[2]) for i in range(self.size_ini)]

                for i in range(self.size_ini):
                    for j in range(3):
                        for k in range(3):
                            if any(self.basis_hessian[i][j][k].l > self.lmax):
                                self.lmax = np.max(self.basis_hessian[i][j][k].l)

                self.n_sph_harm = (self.lmax + 1) ** 2
                self.size = self.npart * self.n_sph_harm
                self.sized = True

                # The basis set is a list of single spherical wave function symbols spanning all the particles
                self.basis = [swf.sph_wf_symbol(1, self.jlm_to_index(i)[1], self.jlm_to_index(i)[2])
                              for i in range(self.size)]

                # Redefine the basis hessian for the new size
                self.basis_hessian = [swf.sph_wf_deriv_tensor(1, self.jlm_to_index(i)[1],
                                                              self.jlm_to_index(i)[2]) for i in range(self.size)]


                # The separation matrix projects basis functions supported by one particle onto the basis functions
                # supported by another particle. The separation matrix is represented as a [self.size]**2 matrix
                self.basis_separation_mat = [[sep.separation_matrix(1, self.jlm_to_index(i)[1], self.jlm_to_index(i)[2],
                                                                    self.jlm_to_index(j)[1], self.jlm_to_index(j)[2])
                                              for i in range(self.size)] for j in range(self.size)]

                # The distance matrix represents the separation between particles.
                if isinstance(kwargs["part"], list):
                    self.distance_mat = [[self.part[self.jlm_to_index(i)[0]].pos
                                          - self.part[self.jlm_to_index(j)[0]].pos for i in range(self.size)]
                                         for j in range(self.size)]
                else:
                    self.distance_mat = [[self.part.pos
                                          - self.part.pos for i in range(self.size)]
                                         for j in range(self.size)]
            else:
                raise TypeError("Unsupported basis set type")
        except KeyError:
            print("Missing basis set parameters")
            print(kwargs.keys())

    def __call__(self, *args, **kwargs):
        # TODO: Unit testing call in basis_set
        if 'type' in kwargs:
            func_type = kwargs['type']
        else:
            func_type = 'regular'

        if func_type == 'regular':
            if len(args) == 3:
                i = self.jlm_to_index(args[0], args[1], args[2])
                return self.basis[i]
            elif len(args) == 1:
                return self.basis[args[0]]
        elif func_type == 'deriv':
            if len(args) == 3:
                i = self.jlm_to_index(args[0], args[1], args[2])
                return self.basis_hessian[i]
            elif len(args) == 1:
                return self.basis_hessian[args[0]]
        else:
            raise KeyError("Undefined basis function type. please choose either regular or deriv")

    def hessian_proj(self, index, i, j):
        val = np.zeros(self.size, dtype=complex)
        for k in range(self.size):
            for kp in range(self.basis_hessian[index][i][j].length):
                if self.basis[k].l == self.basis_hessian[index][i][j].l[kp] \
                        and self.basis[k].m == self.basis_hessian[index][i][j].m[kp]:
                    val[k] = val[k] + self.basis_hessian[index][i][j].a[kp]
        return val

    def jlm_to_index(self, *args):
        # TODO: Unit testing jlm_to_index in basis_set
        if self.sized:
            size = self.size
        else:
            size = self.size_ini

        if len(args) == 3:
            if args[1] > self.lmax or args[0] > self.npart:
                raise ValueError("l or particle Index out of range")
            elif abs(args[2]) > args[1]:
                raise ValueError("|m| > l")
            else:
                return args[0] * (self.lmax + 1) ** 2 + (args[1]) ** 2 + args[1] + args[2]
        elif len(args) == 1:
            if args[0] > size:
                raise ValueError("Index out of range")
            j = args[0] // (self.lmax + 1) ** 2
            a = args[0] - j * (self.lmax + 1) ** 2
            l = int(np.floor(a ** 0.5))
            m = a - l ** 2 - l
            return [j, l, m]

    def med_sph_wf_ovlp(self, f):
        # TODO: Unit testing med_sph_wf_ovlp in basis_set
        # Eq. (7c) in Ref 1
        if isinstance(self.part, list):
            return [swf.med_sph_wf_ovlp(self.basis[i], self.part[self.jlm_to_index(i)[0]], f)
                    for i in range(self.size)]
        else:
            return [swf.med_sph_wf_ovlp(self.basis[i], self.part, f)
                    for i in range(self.size)]

    def overlap_matrix(self, f1, f2, eps_i=1):
        ovlp = np.zeros((self.size, self.size), dtype=complex)
        if isinstance(self.part, list):
            for i in range(self.size):
                for j in range(self.size):
                    if self.basis[i].l == self.basis[j].l and self.basis[i].m == self.basis[j].m:
                        ovlp[i, j] = eps_i * swf.sph_wf_ovlp(self.basis[i], self.part[self.jlm_to_index(i)[0]], f1, f2)
        else:
            for i in range(self.size):
                for j in range(self.size):
                    if self.basis[i].l == self.basis[j].l and self.basis[i].m == self.basis[j].m:
                        ovlp[i, j] = eps_i * swf.sph_wf_ovlp(self.basis[i], self.part, f1, f2)
        return ovlp

    def sph_wf_norm(self, functype, f):
        # TODO: Unit testing sph_wf_norm in basis_set
        if self.npart == 1:
            val = [self.basis[i].norm(f, self.part, functype=functype)
                   for i in range(self.size)]
        else:
            val = [self.basis[i].norm(f, self.part[self.jlm_to_index(i)[0]], functype)
                   for i in range(self.size)]

        return val

    def basis_fun_integral(self, f):
        k = self.part.k(f)
        r = self.part.R
        value = np.zeros(self.size_ini, dtype=complex)
        for i in range(self.size_ini):
            value[i] = self.basis[i].integral(k, r)

        return value