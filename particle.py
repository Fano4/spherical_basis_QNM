import numpy as np
import mathfunctions

import material


class particle:
    def __init__(self, pos, R, mat: material):
        self.R = R  # Radius
        self.pos = pos  # Position of the particle in cartesian coordinate
        self.mat = mat  # Material structure
        pass

    def cart_sph_cen_coord(self, r):
        """Transforms a cartesian coordinate vector to spherical coordinates cnetered on the particle"""
        x = np.stack([r[0] - self.pos[0], r[1] - self.pos[1], r[2] - self.pos[2]])
        sph = np.zeros(x.shape)
        mathfunctions.pcart_to_spher(x[0], x[1], x[2], sph[0], sph[1], sph[2])
        return sph

    def inout(self, r):
        return self.R ** 2 < (r[0] - self.pos[0]) ** 2 + (r[1] - self.pos[1]) ** 2 + (r[2] - self.pos[2]) ** 2

    def k(self,f):
        return self.mat.k(f)
