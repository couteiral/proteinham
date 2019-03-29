import math
import numpy as np
import sympy as sp
from tqdm import tqdm
from copy import deepcopy
from functools import reduce

from qlogic import *
from int_matrix import int_matrix

import sys
sys.path.append('../core/')
from hamiltonian import Hamiltonian

class BeadHamiltonian2D(Hamiltonian):

    is_Bead = True
    is_2D   = True

    def __init__(self, pepstring):
        """Encapsulates the expression and methods of
        a protein hamiltonian of the "bead encoding" form,
        described by Perdomo et al., 2008."""

        self.pepstring = pepstring
        self.naas      = len(pepstring)
        self.nbpd      = math.ceil( math.log2(self.naas) )
        self.dim       = 2
        self.n_bits    = self.dim * self.nbpd * self.naas
        self.int_mat   = int_matrix(pepstring)

        self.bit_list  = [
            sp.Symbol('q_{:d}'.format(i+1), idempotent=True)
            for i in range(self.n_bits)
        ]

        self.expr      = (self.naas+1) * self.steric_term()
        self.expr     += self.naas * self.primary_structure_term()
        self.expr     += self.interaction_term()
        self.expr      = sp.expand(self.expr)

    def pointer(self, i, k):
        """Returns the index of the first bit   
        representing the kth coordinate of the 
        ith residue."""
        return self.nbpd * (2*(i-1) + (k-1))

    def get(self, k):
        """Access the kth bit of the hamiltonian."""
        return self.bit_list[k]

    def _steric_term(self, i, j):
        """Implements the steric interaction between 
        the ith and jth residues.""" 

        return qand([
            qand([
                qxnor(self.get(self.pointer(i, k) + r),
                      self.get(self.pointer(j, k) + r))
            for r in range(self.nbpd)])
        for k in range(1, self.dim+1)])

    def steric_term(self):
        """Implements the steric part of the hamiltonian."""
    
        return sum([
            sum([
                self._steric_term(i, j)
            for j in range(i+1, self.naas+1)])
        for i in range(1, self.naas)])
   
    def dist(self, i, j):
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
            self.dist(i, i+1)
        for i in range(1, self.naas)]) - (self.naas-1)

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
                     qnot(self.get(self.pointer(i, 1))),
                     self.get(self.pointer(j, 1))
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
            [
                qand([
                    qxnor(self.get(self.pointer(i, 3) + s),
                          self.get(self.pointer(j, 3) + s))
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
            ]
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
            ]
            [
                qand([
                    qxnor(self.get(self.pointer(i, 3) + s),
                          self.get(self.pointer(j, 3) + s))
                for s in range(self.nbpd)])
            ]
        )

    def pairwise(self, i, j):
        """Implements the pairwisde interaction term
        between the ith and jth residues."""
    
        return  self.x_plus(i, j) +  self.y_plus(i, j) +  \
                self.x_minus(i, j) + self.y_minus(i, j)

    def interaction_term(self):
        """Contact interaction terms."""
    
        return -1 * sum([
            sum([
                self.int_mat[i-1, j-1] * self.pairwise(i, j)
            for j in range(1, self.naas+1)])
        for i in range(1, self.naas+1)])


class BeadHamiltonian3D(Hamiltonian):

    is_Bead = True
    is_3D   = True

    def __init__(self, pepstring):
        """Encapsulates the expression and methods of
        a protein hamiltonian of the "bead encoding" form,
        described by Perdomo et al., 2008."""

        self.naas     = len(pepstring)
        self.nbpd     = math.ceil( math.log2(self.naas) )
        self.dim      = 3
        self.n_bits   = self.dim * self.nbpd * self.naas
        self.int_mat  = int_matrix(pepstring)

        self.bit_list = [
            sp.Symbol('q_{:d}'.format(i+1))
            for i in range(self.n_bits)
        ]

        self.expr     = (self.naas+1) * self.steric_term()
        self.expr    += self.naas * self.primary_structure_term()
        self.expr    += self.interaction_term()
        self.expr     = sp.expand(self.expr)

    def pointer(self, i, k):
        """Returns the index of the first bit   
        representing the kth coordinate of the 
        ith residue."""
        return self.nbpd * (2*(i-1) + (k-1))

    def get(self, k):
        """Access the kth bit of the hamiltonian."""
        return self.bit_list[k]

    def _steric_term(self, i, j):
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
                self._steric_term(i, j)
            for j in range(i+1, self.naas+1)])
        for i in range(1, self.naas)])
   
    def dist(self, i, j):
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
            self.dist(i, i+1)
        for i in range(1, self.naas)]) - (self.naas-1)

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
                     qnot(self.get(self.pointer(i, 1))),
                     self.get(self.pointer(j, 1))
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

    def pairwise(self, i, j):
        """Implements the pairwise interaction term
        between the ith and jth residues."""
    
        return  self.x_plus(i, j) +  self.y_plus(i, j) +  \
                self.x_minus(i, j) + self.y_minus(i, j)

    def interaction_term(self):
        """Contact interaction terms."""
    
        return -1 * sum([
            sum([
                self.int_mat[i-1, j-1] * self.pairwise(i, j)
            for j in range(1, self.naas+1)])
        for i in range(1, self.naas+1)])

