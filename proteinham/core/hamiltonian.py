"""

Base class for hamiltonians.

Carlos Outeiral
March 2019

"""

import sympy as sp
from abc import *
from copy import deepcopy

from int_matrix import int_matrix


class Hamiltonian(ABC):

    is_Bead        = False
    is_Diamond     = False
    is_TurnAncilla = False
    is_TurnCircuit = False
    is_TurnOneHot  = False

    is_2D          = False
    is_3D          = False


    def _proc_input(self, pepstring):
        self.pepstring = pepstring
        self.naas      = len(pepstring)
        self.int_mat   = int_matrix(pepstring)

    def _create_bitreg(self):
        self.bit_list  = [
            sp.Symbol('q_%d' % (i+1), idempotent=True)
            for i in range(self.n_bits)
        ]

    @abstractmethod
    def build_exp(self):
        pass

    def get(self, i):
        """Return the ith bit of the hamiltonian."""
        return self.bit_list[i]

    def to_circuit(self):
        circuit = deepcopy(self.expr)
        for i in range(self.n_bits):
            circuit = circuit.subs(self.bit_list[i],
                                   1-sp.Symbol('Z_%d' % (i+1)) )

        return sp.expand(circuit)

