import math
import numpy as np
import sympy as sp
import symengine as se
from abc import *
from tqdm import tqdm
from copy import deepcopy
from functools import reduce

from proteinham.core.hamiltonian import Hamiltonian
from .qlogic import *


class CommonBeadHamiltonian(Hamiltonian):

    is_Bead = True

    def __init__(self, pepstring):
        """Encapsulates the expression and methods of
        a protein hamiltonian of the "bead encoding" form,
        described by Perdomo et al., 2008."""

        self._proc_input(pepstring)
        self.nbpd   = math.ceil( math.log2(len(pepstring)) )
        self.n_bits = self.dim * self.nbpd * self.naas
        self._create_bitreg()
        self.build_exp()

    @property
    def encoding(self):
        return 'bead'

    def build_exp(self):
        self.expr      = (self.naas+1) * self.steric_term()
        self.expr     += self.naas * self.primary_structure_term()
        self.expr     += self.interaction_term()
        self.expr      = se.expand(self.expr)
        self.n_terms   = len(self.expr.args)

    def pointer(self, i, k):
        """Returns the index of the first bit   
        representing the kth coordinate of the 
        ith residue.

        WARNING: both i and k start in 1"""

        if k > self.dim or i > self.naas or k <= 0 or i <= 0:
            return ValueError('out of bounds')
        return self.nbpd * (self.dim*(i-1) + (k-1))

    def steric_term_ij(self, i, j):
        """Implements the steric interaction between 
        the ith and jth residues.""" 

        return qand([
            qand([
                qxnor(self.get(self.pointer(i, k) + r),
                      self.get(self.pointer(j, k) + r))
            for r in range(self.nbpd)])
        for k in range(1, self.dim+1)])

    def steric_term(self):
        """Implements the steric interaction."""
    
        return sum([
            sum([
                self.steric_term_ij(i, j)
            for j in range(i+1, self.naas+1)])
        for i in range(1, self.naas)])
   
    def l1_dist_ij(self, i, j):
        """Implements the L1-distance between the
        ith and jth residues."""
    
        return sum([
            sum([2**r * (self.get(self.pointer(i, k) + r) -
                         self.get(self.pointer(j, k) + r))
            for r in range(self.nbpd)])**2
        for k in range(1, self.dim+1)])
   
    def primary_structure_term(self):
        """Implements the primary structure part of the
        hamiltonian."""
    
        return sum([
            self.l1_dist_ij(i, i+1)
        for i in range(1, self.naas)]) - (self.naas-1)

    def interaction_term(self):
        """Contact interaction terms."""
    
        return -1 * sum([
            sum([
                self.int_mat[i-1, j-1] * self.interaction_term_ij(i, j)
            for j in range(1, self.naas+1)])
        for i in range(1, self.naas+1)])

    @property
    @abstractmethod
    def dim(self):
        pass

    @abstractmethod
    def interaction_term_ij(self):
        pass


class BeadHamiltonian2D(CommonBeadHamiltonian):

    is_2D   = True

    @property
    def dim(self):
        return 2

    def x_plus(self, i, j):
        """Implements the term x+ for pairwise interactions."""
    
        return qand(
            [
                qand([
                     qnot(self.get(self.pointer(i, 1))),
                     self.get(self.pointer(j, 1))
                ])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 1) + s),
                          self.get(self.pointer(j, 1) + s))
                for s in range(1, self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 2) + s),
                          self.get(self.pointer(j, 2) + s))
                for s in range(self.nbpd)])
            ]
         )
   
    def y_plus(self, i, j):
        """Implements the term y+ for pairwise interactions."""
    
        return qand(
            [
                qand([
                     qnot(self.get(self.pointer(i, 2))),
                     self.get(self.pointer(j, 2))
                ])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 2) + s),
                          self.get(self.pointer(j, 2) + s))
                for s in range(1, self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 1) + s),
                          self.get(self.pointer(j, 1) + s))
                for s in range(1, self.nbpd)])
            ]
         )
   
    def x_minus(self, i, j):
        """Implements the term x- for pairwise interactions."""
    
        return qand(
            [
                qand([
                    qnot(self.get(self.pointer(i, 1))),
                    self.get(self.pointer(j, 1))
                ])
            ] +
            [
                1 - qand([qnot(self.get(self.pointer(i, 1) + k))
                for k in range(self.nbpd)])
            ] +
            [
                qxnor(
                    self.get(self.pointer(i, 1) + 1),
                    qnot(self.get(self.pointer(j, 1) + 1))
                )
            ] +
            [
                qand([
                    qxnor(
                        qnot(
                            self.get(self.pointer(i, 1) + r) +
                            qand([
                                self.get(self.pointer(i, 1) + u)
                            for u in range(1, r-1)]) \
                            -2 * qand([
                                self.get(self.pointer(i, 1) + u)
                            for u in range(1, r)])
                        ),
                        self.get(self.pointer(i, 1) + r)
                    )
                for r in range(2, self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 2) + s),
                          self.get(self.pointer(j, 2) + s))
                for s in range(self.nbpd)])
            ]
        )
    
    def y_minus(self, i, j):
        """Implements the term y- for pairwise interactions."""
    
        return qand(
            [
                qand([
                    qnot(self.get(self.pointer(i, 2))),
                    self.get(self.pointer(j, 2))
                ])
            ] +
            [
                1 - qand([qnot(self.get(self.pointer(i, 2) + k))
                for k in range(self.nbpd)])
            ] +
            [
                qxnor(
                    self.get(self.pointer(i, 2) + 1),
                    qnot(self.get(self.pointer(j, 2) + 1))
                )
            ] +
            [
                qand([
                    qxnor(
                        qnot(
                            self.get(self.pointer(i, 2) + r) +
                            qand([
                                self.get(self.pointer(i, 2) + u)
                            for u in range(1, r-1)]) \
                            -2 * qand([
                                self.get(self.pointer(i, 2) + u)
                            for u in range(1, r)])
                        ),
                        self.get(self.pointer(i, 2) + r)
                    )
                for r in range(2, self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 1) + s),
                          self.get(self.pointer(j, 1) + s))
                for s in range(self.nbpd)])
            ]
        )

    def interaction_term_ij(self, i, j):
        """Implements the pairwise interaction term
        between the ith and jth residues."""
    
        return  self.x_plus(i, j) + self.x_minus(i, j) + \
                self.y_plus(i, j) + self.y_minus(i, j)



class BeadHamiltonian3D(CommonBeadHamiltonian):

    is_3D   = True

    @property
    def dim(self):
        return 3

    def x_plus(self, i, j):
        """Implements the term x+ for pairwise interactions."""

        return qand(
            [
                qand([
                     qnot(self.get(self.pointer(i, 1))),
                     self.get(self.pointer(j, 1))
                ])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 1) + s),
                          self.get(self.pointer(j, 1) + s))
                for s in range(1, self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 2) + s),
                          self.get(self.pointer(j, 2) + s))
                for s in range(self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 3) + s),
                          self.get(self.pointer(j, 3) + s))
                for s in range(self.nbpd)])
            ]
         )
   
    def y_plus(self, i, j):
        """Implements the term y+ for pairwise interactions."""
    
        return qand(
            [
                qand([
                     qnot(self.get(self.pointer(i, 2))),
                     self.get(self.pointer(j, 2))
                ])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 2) + s),
                          self.get(self.pointer(j, 2) + s))
                for s in range(1, self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 1) + s),
                          self.get(self.pointer(j, 1) + s))
                for s in range(1, self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 3) + s),
                          self.get(self.pointer(j, 3) + s))
                for s in range(self.nbpd)])
            ]
         )
   
    def z_plus(self, i, j):
        """Implements the term z+ for pairwise interactions."""
    
        return qand(
            [
                qand([
                     qnot(self.get(self.pointer(i, 3))),
                     self.get(self.pointer(j, 3))
                ])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 3) + s),
                          self.get(self.pointer(j, 3) + s))
                for s in range(1, self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 1) + s),
                          self.get(self.pointer(j, 1) + s))
                for s in range(1, self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 2) + s),
                          self.get(self.pointer(j, 2) + s))
                for s in range(self.nbpd)])
            ]
         )
   
    def x_minus(self, i, j):
        """Implements the term x- for pairwise interactions."""
    
        return qand(
            [
                qand([
                    qnot(self.get(self.pointer(i, 1))),
                    self.get(self.pointer(j, 1))
                ])
            ] +
            [
                1 - qand([qnot(self.get(self.pointer(i, 1) + k))
                for k in range(self.nbpd)])
            ] +
            [
                qxnor(
                    self.get(self.pointer(i, 1) + 1),
                    qnot(self.get(self.pointer(j, 1) + 1))
                )
            ] +
            [
                qand([
                    qxnor(
                        qnot(
                            self.get(self.pointer(i, 1) + r) +
                            qand([
                                self.get(self.pointer(i, 1) + u)
                            for u in range(1, r-1)]) \
                            -2 * qand([
                                self.get(self.pointer(i, 1) + u)
                            for u in range(1, r)])
                        ),
                        self.get(self.pointer(i, 1) + r)
                    )
                for r in range(2, self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 2) + s),
                          self.get(self.pointer(j, 2) + s))
                for s in range(self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 3) + s),
                          self.get(self.pointer(j, 3) + s))
                for s in range(self.nbpd)])
            ]
        )
    
    def y_minus(self, i, j):
        """Implements the term y- for pairwise interactions."""
    
        return qand(
            [
                qand([
                    qnot(self.get(self.pointer(i, 2))),
                    self.get(self.pointer(j, 2))
                ])
            ] +
            [
                1 - qand([qnot(self.get(self.pointer(i, 2) + k))
                for k in range(self.nbpd)])
            ] +
            [
                qxnor(
                    self.get(self.pointer(i, 2) + 1),
                    qnot(self.get(self.pointer(j, 2) + 1))
                )
            ] +
            [
                qand([
                    qxnor(
                        qnot(
                            self.get(self.pointer(i, 2) + r) +
                            qand([
                                self.get(self.pointer(i, 2) + u)
                            for u in range(1, r-1)]) \
                            -2 * qand([
                                self.get(self.pointer(i, 2) + u)
                            for u in range(1, r)])
                        ),
                        self.get(self.pointer(i, 2) + r)
                    )
                for r in range(2, self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 1) + s),
                          self.get(self.pointer(j, 1) + s))
                for s in range(self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 3) + s),
                          self.get(self.pointer(j, 3) + s))
                for s in range(self.nbpd)])
            ]
        )

    def z_minus(self, i, j):
        """Implements the term z- for pairwise interactions."""
    
        return qand(
            [
                qand([
                    qnot(self.get(self.pointer(i, 3))),
                    self.get(self.pointer(j, 3))
                ])
            ] +
            [
                1 - qand([qnot(self.get(self.pointer(i, 3) + k))
                for k in range(self.nbpd)])
            ] +
            [
                qxnor(
                    self.get(self.pointer(i, 3) + 1),
                    qnot(self.get(self.pointer(j, 3) + 1))
                )
            ] +
            [
                qand([
                    qxnor(
                        qnot(
                            self.get(self.pointer(i, 3) + r) +
                            qand([
                                self.get(self.pointer(i, 3) + u)
                            for u in range(1, r-1)]) \
                            -2 * qand([
                                self.get(self.pointer(i, 3) + u)
                            for u in range(1, r)])
                        ),
                        self.get(self.pointer(i, 3) + r)
                    )
                for r in range(2, self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 1) + s),
                          self.get(self.pointer(j, 1) + s))
                for s in range(self.nbpd)])
            ] +
            [
                qand([
                    qxnor(self.get(self.pointer(i, 2) + s),
                          self.get(self.pointer(j, 2) + s))
                for s in range(self.nbpd)])
            ]
        )
    
    def interaction_term_ij(self, i, j):
        """Implements the pairwisde interaction term
        between the ith and jth residues."""
   
        return  self.x_plus(i, j) + self.x_minus(i, j) + \
                self.y_plus(i, j) + self.y_minus(i, j) + \
                self.z_plus(i, j) + self.z_minus(i, j) 
