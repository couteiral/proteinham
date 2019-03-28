import math
import numpy as np
import sympy as sp
from qlogic import *
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from functools import reduce
from int_matrix import int_matrix


class DiamondHamiltonian2D(object):

    is_Diamond = True
    is_2D      = True

    def __init__(self, pepstring):
        """Encapsulates the expression and methods of
        a protein hamiltonian of the "diamond encoding" form,
        described by Babbush et al., 2012."""

        self.naas     = len(pepstring)
        self.n_bits   = sum([self.n_k(k) for k in range(self.naas)])
        self.dim      = 2
        self.int_mat  = int_matrix(pepstring)

        self.bit_list = [
            sp.Symbol('q_{:d}'.format(i+1), idempotent=True)
            for i in range(self.n_bits)
        ]
        
        self.expr     = self.one_term()
        self.expr    += self.primary_structure_term()
        self.expr    += self.steric_term()
        self.expr    += self.interaction_term()
        self.expr     = sp.expand(self.expr)

    def pointer(self, i):
        """Returns the index of the first bit in
        the string representing the ith residue."""
        return sum([self.n_k(t) for t in range(i)])

    def get(self, k):
        """Access the kth bit of the hamiltonian."""
        return self.bit_list[k]

    def k_rhombus(self, k):
        """Returns the coordinates of each position
        in the kth rhombus."""
    
        rhombus = list()
        for t in range(-k, 0):
            rhombus.append((-t, k+t))
        for t in range(0, k):
            rhombus.append((-t, k-t))
        for t in range(-k, 0):
            rhombus.append((t, -k-t))
        for t in range(0, k):
            rhombus.append((t, -k+t))
        return rhombus
    
    def rhombus(self, k):
        """Returns the coordinates of each position
        that the kth residue may occupy."""
    
        if k == 0: return None
        if k <= 2: return self.k_rhombus(k)
        if k % 2 == 0:
            return list(chain(*[self.k_rhombus(r) for r in range(2, k+2, 2)]))
        else:
            return list(chain(*[self.k_rhombus(r) for r in range(1, k+2, 2)]))
    
    def n_k(self, k):
        """Returns the number of bits necessary to
        encode the kth residue."""
    
        if k == 0:
            return 0
        else:
            return len(self.rhombus(k))
    
    def adjacency(self, i, k, j, l):
        """Returns 1 if q_i^k and q_j^l are adjacent
        and 0 otherwise."""
    
        rhombus_k = self.rhombus(k)
        rhombus_l = self.rhombus(l)
    
        qik = rhombus_k[i]
        qjl = rhombus_l[j]
        dist = math.sqrt((qik[0]-qjl[0])**2 + \
                         (qik[1]-qjl[1])**2)
    
        if dist == 1.0:
            return 1
        else:
            return 0
    
    def one_term(self):
        """Implements the term tthat avoids multiple
        bits turned on for the same residue."""
    
        return sum([
            sum([
                sum([
                    self.get(self.pointer(k)+i) * \
                    self.get(self.pointer(k)+j)
                for j in range(i+1, self.n_k(k))])
            for i in range(self.n_k(k)-1)])
        for k in range(1, self.naas)])
    
    def primary_structure_term(self):
        """Implements the primary structure part of the
        hamiltonian."""
    
        return  (self.naas-2) +\
        sum([
            sum([
                sum([
                    self.get(self.pointer(k)   + i) * \
                    self.get(self.pointer(k-1) + j) * \
                    self.adjacency(i, k, j, k-1)
                for j in range(self.n_k(k-1))])
            for i in range(self.n_k(k))])
        for k in range(2, self.naas)])
    
    def steric_term(self):
        """Implements the steric part of the hamiltonian."""
    
        return sum([
            sum([
                sum([
                    ((1+k-h) % 2) * \
                    self.get(self.pointer(k)+i) * \
                    self.get(self.pointer(h)+i)
                for i in range(self.n_k(k))])
            for h in range(k+1, self.naas)])
        for k in range(1, self.naas-1)])
    
    def interaction_term(self):
        """Implements the pairwise interaction term
        between the ith and jth residues."""
    
        expr = sp.numbers.Integer(0)
    
        # First we handle the 0th residue that is not encoded
        # directly
        for h in range(2, self.naas):
    
            if self.int_mat[0, h] == 0: continue
            if (h % 2) == 0: continue
            rhombus_h = self.rhombus(h)
    
            for i in range(self.n_k(h)):
    
                qih = rhombus_h[i]
                dist = math.sqrt(qih[0]**2 + qih[1]**2)
                if dist == 1.0:
                    expr += self.int_mat[0, h] * \
                            self.get(self.pointer(h)+i)
    
        # Rest of residues
        for k in range(1, self.naas):
            for h in range(k+1, self.naas):
    
                if self.int_mat[k, h] == 0: continue
                if ((k-h) % 2) == 0: continue
    
                for i in range(self.n_k(k)):
                    for j in range(self.n_k(h)):
    
                        if not adjacency(i, k, j, h): continue
    
                        expr += self.int_mat[h, k] * \
                                self.get(self.pointer(k)+i) * \
                                self.get(self.pointer(h)+j)
    
        return expr
                       
