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


