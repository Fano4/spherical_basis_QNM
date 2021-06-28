import numpy as np
import spherical_wave_function as swf


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

                self.basis = [[[swf.sph_wf_symbol(1,l,m) for m in np.arange(-l,l+1)] for l in range(self.lmax+1)]
                                for _ in range(self.npart)]
                self.basis_gradient = [[[swf.sph_wf_deriv_tensor(1,l,m) for m in np.arange(-l,l+1)] for l in range(self.lmax+1)]
                                       for _ in range(self.npart)]

            else:
                raise TypeError("Unsupported basis set type")
        except KeyError:
            print("Missing basis set parameters")
            print(kwargs.keys())