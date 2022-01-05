# This file implements a class for the electric field in the basis of spherical wave functions.
# The field vector is defined in the basis of spherical waves as a function of frequency

import numpy as np


class field:
    """The field class implements the electric field representation in the spherical waves basis

    The field is defined as a function of frequency and allows to use either built-in or custom functions.
    The default field type is a classical field of complex frequency freq_cent
    The lorentzian field has a lorentzian spectrum
    The field type must be set when initializing the class.

    init arguments:
        self._init_(self, sph_coeff, [type =, freq_cent =, amplitude =, width =, custom_func=])
        type = ['classical','custom']
    """

    def __init__(self, *args, **kwargs):
        # TODO: Unit test constructor in field class
        self.sph_coeff = args[0]
        if 'type' in kwargs.keys():
            self.type = kwargs['type']
        else:
            print("Using default classical field type")
            self.type = 'classical'

        if self.type == 'classical':
            if isinstance(kwargs['freq_cent'], np.ndarray):
                self.freq = kwargs['freq_cent']
            else:
                raise TypeError("Expected a numpy array for central frequency of each mode.")

        elif self.type == 'lorentzian':
            if isinstance(kwargs['freq_cent'], np.ndarray) and (kwargs['width'], np.array):
                self.freq_cent = kwargs['freq_cent']
                self.width = kwargs['width']
            else:
                raise TypeError("Expected a numpy array for central frequency and width of each mode with lorentzian "
                                "type.")

        elif self.type == 'custom':
            self.custom_func = kwargs['custom_func']

        return

    def __call__(self, f: float):
        # TODO: Unit test call in field class
        if self.type == 'classical':
            if isinstance(self.freq, float):
                return float(self.freq == f)
            elif isinstance(self.freq, complex):
                return (2 / np.pi) ** 0.5 * np.imag(self) / ((f - np.real(self)) ** 2 + np.imag(self) ** 2)

        elif self.type == 'lorentzian':
            return (2 / np.pi) ** 0.5 * (self.width / 2) / ((f - self.freq_cent) ** 2 + (self.width / 2) ** 2)

        elif self.type == 'custom':
            return self.custom_func(f)

        else:
            raise ValueError("Unrecognized field type.")
