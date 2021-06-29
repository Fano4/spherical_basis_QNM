import basis_set
import spherical_wave_function as swf


class green_function:
    def __init__(self,basis_set):
        self.basis_set = basis_set
        for j in range(basis_set.npart):
            for jp in range(basis_set.npart):
                if j == jp:
                    #Self term block
                    self.self_block()
                else:
                    # Scattering term block
                    self.scattering_block(basis_set)
                    return
        return

    def self_block(self):
        return
    def scattering_block(self, basis_set: basis_set.basis_set):

        size = basis_set.size
        return