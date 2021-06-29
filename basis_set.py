import numpy as np
import spherical_wave_function as swf
import separation_matrix as sep


class basis_set:
    def __init__(self,**kwargs):
        try:
            if kwargs['type'] == 'spherical':

                # Start by setting up the basis functions
                self.lmax = kwargs['lmax']
                if isinstance(kwargs['part'],list):
                    self.npart = len(kwargs['part'])
                else:
                    self.npart = 1
                self.part = kwargs['part']

                self.size = self.npart * (self.lmax + 1)**2

                self.basis = [swf.sph_wf_symbol(1,self._jlm_to_index(i)[1],self._jlm_to_index(i)[2]) for i in range(self.size)]
                self.basis_hessian = [swf.sph_wf_deriv_tensor(1,self._jlm_to_index(i)[1],
                                                               self._jlm_to_index(i)[2]) for i in range(self.size)]
                self.basis_separation_mat = [[sep.separation_matrix(self._jlm_to_index(i)[1], self._jlm_to_index(i)[2],
                                                                     self._jlm_to_index(j)[1], self._jlm_to_index(j)[2])
                                               for i in range(self.size)] for j in range(self.size)]
                self.distance_mat = [[self.part[self._jlm_to_index(i)[0]].pos - self.part[self._jlm_to_index(j)[0]].pos
                                      for i in range(self.size)] for j in range(self.size)]

            else:
                raise TypeError("Unsupported basis set type")
        except KeyError:
            print("Missing basis set parameters")
            print(kwargs.keys())

    def __call__(self, *args,**kwargs):
        if 'type' in kwargs:
            type = kwargs['type']
        else:
            type = 'regular'

        if type == 'regular':
            if len(args) == 3:
                i = self._jlm_to_index(args[0], args[1], args[2])
                return self.basis[i]
            elif len(args) == 1:
                return self.basis[args[0]]
        elif type == 'deriv':
            if len(args) == 3:
                i = self._jlm_to_index(args[0], args[1], args[2])
                return self.basis_hessian[i]
            elif len(args) == 1:
                return self.basis_hessian[args[0]]
        else:
            raise KeyError("Undefined basis function type. please choose either regular or deriv")

    def _jlm_to_index(self,*args):
        if len(args) == 3:
            if abs(args[2]) > args[1]:
                raise ValueError("|m| > l")
            else:
                return args[0] * (self.lmax + 1)**2 + (args[1])**2 + args[1] + args[2]
        elif len(args) == 1:
            j = args[0] // (self.lmax + 1)**2
            a = args[0] - j * (self.lmax + 1)**2
            l = int(np.floor(a ** 0.5))
            m = a - l**2 - l
            return [j, l, m]

    def med_sph_wf_ovlp(self,f):

        return [swf.med_sph_wf_ovlp(self.basis[i].l,
                                    self.part[self._jlm_to_index(i)[0]],
                                    f)
                for i in range(self.size)]

    def sph_wf_norm(self, f, type):
        if type == 'mat':
            return [swf.normalization_cst(self.basis[i].l,
                                          self.part[self._jlm_to_index(i)[0]].k(f),
                                          self.part[self._jlm_to_index(i)[0]].R)
                    for i in range(self.size)]
        elif type == 'background':
            return [swf.normalization_cst(self.basis[i].l,
                                          self.part[self._jlm_to_index(i)[0]].med.k(f),
                                          self.part[self._jlm_to_index(i)[0]].R)
                    for i in range(self.size)]
        else:
            raise ValueError('Unrecognized normalization type.')
