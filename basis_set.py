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
        try:
            self.bas_func = kwargs['type']
            if self.bas_func == 'spherical':
                # Initialize spherical basis functions

                self.lmax = kwargs['lmax']

                if isinstance(kwargs['part'], list):
                    self.npart = len(kwargs['part'])
                    raise ValueError("The method has not been checked for more than one particle")
                else:
                    self.npart = 1

                self.part = kwargs['part']
                self.n_sph_harm = (self.lmax + 1) ** 2
                self.size = self.npart * self.n_sph_harm

                # The basis set is a list of single spherical wave function symbols spanning all the particles
                self.basis = [swf.sph_wf_symbol(1, self.jlm_to_index(i)[1], self.jlm_to_index(i)[2])
                              for i in range(self.size)]

                # The Hessian matrix is a rank-3 tensor of the spherical basis function symbols' second derivatives.
                # Each symbol is a component of the Hessian in cartesian coordinates.
                self.basis_hessian = [swf.sph_wf_deriv_tensor(1, self.jlm_to_index(i)[1],
                                                              self.jlm_to_index(i)[2]) for i in range(self.size)]

                # The separation matrix projects basis functions supported by one particle onto the basis functions
                # supported by another particle. The separation matrix is represented as a [self.size]**2 matrix
                self.basis_separation_mat = [[sep.separation_matrix(self.jlm_to_index(i)[1], self.jlm_to_index(i)[2],
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

    def jlm_to_index(self, *args):
        if len(args) == 3:
            if abs(args[2]) > args[1]:
                raise ValueError("|m| > l")
            else:
                return args[0] * (self.lmax + 1) ** 2 + (args[1]) ** 2 + args[1] + args[2]
        elif len(args) == 1:
            j = args[0] // (self.lmax + 1) ** 2
            a = args[0] - j * (self.lmax + 1) ** 2
            l = int(np.floor(a ** 0.5))
            m = a - l ** 2 - l
            return [j, l, m]

    def med_sph_wf_ovlp(self, f):
        # Eq. (7c) in Ref 1
        return [swf.med_sph_wf_ovlp(self.basis[i].l,
                                    self.part[self.jlm_to_index(i)[0]],
                                    f)
                for i in range(self.size)]

    def sph_wf_norm(self, functype, f):
        if functype == 'mat':
            return [swf.normalization_cst(self.basis[i].l,
                                          self.part[self.jlm_to_index(i)[0]].k(f),
                                          self.part[self.jlm_to_index(i)[0]].R)
                    for i in range(self.size)]
        elif functype == 'background':
            return [swf.normalization_cst(self.basis[i].l,
                                          self.part[self.jlm_to_index(i)[0]].med.k(f),
                                          self.part[self.jlm_to_index(i)[0]].R)
                    for i in range(self.size)]
        else:
            raise ValueError('Unrecognized normalization type. Support types are \'mat\' and \'background\' ')
