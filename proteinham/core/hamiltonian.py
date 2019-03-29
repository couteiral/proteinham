"""

Base class for hamiltonians.

Carlos Outeiral
March 2019

"""

import sympy as sp
from abc import *


class Hamiltonian(ABC):

    is_Bead        = False
    is_Diamond     = False
    is_TurnAncilla = False
    is_TurnCircuit = False
    is_TurnOneHot  = False

    is_2D          = False
    is_3D          = False

    is_mutable     = False

    @property
    @abstractmethod
    def bit_list(self):
        pass

    @property
    @abstractmethod
    def dimension(self):
        pass

    @property
    @abstractmethod
    def encoding(self):
        pass

    @property
    @abstractmethod
    def expr(self):
        pass

    @property
    @abstractmethod
    def int_mat(self):
        pass

    @property
    @abstractmethod
    def naas(self):
        pass

    @property
    @abstractmethod
    def n_bits(self):
        pass

    @property
    @abstractmethod
    def pepstring(self):
        pass

    def make_mutable(self):
        self.is_mutable = True

