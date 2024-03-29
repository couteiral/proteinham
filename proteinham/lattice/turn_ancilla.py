import math
import numpy as np
import sympy as sp
#import symengine as se
from abc import *
from tqdm import tqdm
from copy import deepcopy
from functools import reduce

from .qlogic import *
from proteinham.core.hamiltonian import Hamiltonian

class CommonTurnAncillaHamiltonian(Hamiltonian):

    def __init__(self, pepstring):
        """Encapsulates the expression and methods of
        a protein hamiltonian of the "turn ancilla encoding" 
        form, described by Babbush et al., 2012."""

        self._proc_input(pepstring)
        self.start_bit = None
 
        self.n_bits = self.dim * (self.naas-1)
        self.n_bits += sum([ 
            sum([
                self.mu(i, j)
            for j in range(i+4, self.naas)])
        for i in range(self.naas-4)])
        self.n_bits += sum([
            sum([
                1 if self.int_mat[i, j] != 0 else 0
            for j in range(i+3, self.naas)])
        for i in range(self.naas-3)])
    
        self._create_bitreg()
        #self.build_exp()

    def build_exp(self):   
        self.expr      = (self.naas+1) * self.back_term()
        if self.dim == 3:
            self.expr += (self.naas+1)**2 * self.redun_term()
        self.expr     += (self.naas+1) * self.steric_term()
        self.expr     += self.interaction_term()
        #self.expr      = se.expand(self.expr)
        self.expr      = sp.expand(self.expr)
        self.n_terms   = len(self.expr.args)

    def get(self, k):
        """Access the kth bit of the hamiltonian."""
        return self.bit_list[k]

    @property
    @abstractmethod
    def dim(self):
        pass

class TurnAncillaHamiltonian2D(CommonTurnAncillaHamiltonian):

    is_2D = True

    @property
    def dim(self):
        return 2

    def r_pointer(self, i):
        """Points to the start of the string describing
        the ith turn."""
        if i > self.naas: raise ValueError('There are only %s residues' % self.naas)
        return 2*i-2 if i > 0 else 0
        
    def o_pointer(self, i, j):
        """Points to the start of the string containing
        ancillary bits."""
        return 2*self.naas-2 + sum([
             sum([
                self.mu(m, n)
            for n in range(m+4, self.naas)])
        for m in range(i)]) + \
        sum([
            self.mu(i, n)
        for n in range(i+4, j)])
    
    def i_pointer(self, i, j):
        """Points to the ancilla bit encoding the
        interaction between the ith and jth residues."""
   
        if not self.start_bit: 
            self.start_bit = 2*self.naas-2 + sum([
                sum([
                    self.mu(i, j)
                for j in range(i+4, self.naas)])
            for i in range(self.naas-4)])
    
        return self.start_bit + \
        sum([
            sum([ 
                1 if self.int_mat[m, n] != 0 else 0
            for n in range(m+3, self.naas)])
        for m in range(i-1)]) + \
        sum([
            1 if self.int_mat[i, n] != 0 else 0
        for n in range(i+3, j)])
    
    def circuit_xp(self, q_i, q_j):
        """Implements a circuit that returns 1
        if the chain moves in the direction x+."""
        return (1-q_i)*q_j
    
    def circuit_xn(self, q_i, q_j):
        """Implements a circuit that returns 1
        if the chain moves in the direction x-."""
        return q_i*(1-q_j)
    
    def circuit_yp(self, q_i, q_j):
        """Implements a circuit that returns 1
        if the chain moves in the direction y+."""
        return q_i*q_j
    
    def circuit_yn(self, q_i, q_j):
        """Implements a circuit that returns 1
        if the chain moves in the direction y-."""
        return (1-q_i)*(1-q_j)
    
    def x_position(self, n):
        """Computes the x coordinate of the nth residue."""

        if n > self.naas: 
            raise ValueError('n greater than number of residues') 
        return sum([
            self.circuit_xp(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1)) - \
            self.circuit_xn(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1))
        for i in range(n)]) 
    
    def y_position(self, n):
        """Computes the y coordinate of the nth residue."""
    
        if n > self.naas: 
            raise ValueError('n greater than number of residues') 
        return sum([
            self.circuit_yp(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1)) - \
            self.circuit_yn(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1))
        for i in range(n)])
    
    def g(self, i, j):
        """Computes the distance between residues i and j."""

        return (self.x_position(i) - self.x_position(j))**2 \
             + (self.y_position(i) - self.y_position(j))**2
    
    def mu(self, i, j):
        """Computes \mu_{ij}."""
    
        if i == j:
            return 0
        elif abs(i-j) < 3:
            return 0
        else:
            return 2 * int(math.ceil(math.log2(abs(i-j)))) \
                     * ((1+i-j) % 2)
    
    def alpha(self, i, j):
        """Computes \alpha_{ij}."""
        return sum([
            2**k * self.get(self.o_pointer(i, j) + k)
        for k in range(self.mu(i,j))])
    
    def back_term(self):
        """Ensures that the chain does not go
        back on itself."""
    
        return sum([
            self.circuit_xp(self.get(self.r_pointer(i)), 
                            self.get(self.r_pointer(i)+1)) *
            self.circuit_xn(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1)) + \
            self.circuit_xn(self.get(self.r_pointer(i)), 
                            self.get(self.r_pointer(i)+1)) *
            self.circuit_xp(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1)) + \
            self.circuit_yp(self.get(self.r_pointer(i)), 
                            self.get(self.r_pointer(i)+1)) *
            self.circuit_yn(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1)) + \
            self.circuit_yn(self.get(self.r_pointer(i)), 
                            self.get(self.r_pointer(i)+1)) *
            self.circuit_yp(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1))
        for i in range(self.naas-1)])
    
    def steric_term(self):
        """Ensures that the chain does not overlap."""
    
        term = sp.numbers.Integer(0)
        for i in range(self.naas-4):
            for j in range(i+4, self.naas):
                if (1+i-j) % 2:
                    term += (2**self.mu(i, j) -self.g(i, j) \
                         -  self.alpha(i, j))**2
    
        return term

    def interaction_term_ij(self, i, j):
        return -1 * self.get(self.i_pointer(i, j)) \
                           * self.int_mat[i, j] \
                           * ( 2 - self.g(i, j) )
    
    def interaction_term(self):
        """Computes contacts between residues."""
    
        term = sp.numbers.Integer(0)
        for i in range(self.naas-3):
            for j in range(i+3, self.naas):
                if self.int_mat[i, j] == 0: continue
                term -= self.get(self.i_pointer(i, j)) \
                      * self.int_mat[i, j] \
                      * ( 2 - self.g(i, j) )
        return term


class TurnAncillaHamiltonian3D(CommonTurnAncillaHamiltonian):

    is_3D = True

    @property
    def dim(self):
        return 3

    def r_pointer(self, i):
        """Points to the start of the string describing
        the ith turn."""
        return 3*i-3 if i > 0 else 0
        
    def o_pointer(self, i, j):
        """Points to the start of the string containing
        ancillary bits."""
        return 3*self.naas-3 + sum([
             sum([
                self.mu(m, n)
            for n in range(m+4, self.naas)])
        for m in range(i)]) + \
        sum([
            self.mu(i, n)
        for n in range(i+4, j)])
    
    def i_pointer(self, i, j):
        """Points to the ancilla bit encoding the
        interaction between the ith and jth residues."""
   
        if not self.start_bit: 
            self.start_bit = 3*self.naas-3 + sum([
                sum([
                    self.mu(i, j)
                for j in range(i+4, self.naas)])
            for i in range(self.naas-4)])
    
        return self.start_bit + \
        sum([
            sum([ 
                1 if self.int_mat[m, n] != 0 else 0
            for n in range(m+3, self.naas)])
        for m in range(i-1)]) + \
        sum([
            1 if self.int_mat[i, n] != 0 else 0
        for n in range(i+3, j)])

    def circuit_xp(self, q_i, q_j, q_k):
        """Implements a circuit that returns 1
        if the chain moves in the direction x+."""
        return q_i * q_j * q_k

    def circuit_xn(self, q_i, q_j, q_k):
        """Implements a circuit that returns 1
        if the chain moves in the direction x-."""
        return q_i * (1-q_j) * (1-q_k)

    def circuit_yp(self, q_i, q_j, q_k):
        """Implements a circuit that returns 1
        if the chain moves in the direction y+."""
        return q_i * (1-q_j) * q_k

    def circuit_yn(self, q_i, q_j, q_k):
        """Implements a circuit that returns 1
        if the chain moves in the direction y-."""
        return q_i * q_j * (1-q_k)

    def circuit_zp(self, q_i, q_j, q_k):
        """Implements a circuit that returns 1
        if the chain moves in the direction z+."""
        return (1-q_i) * (1-q_j) * q_k

    def circuit_zn(self, q_i, q_j, q_k):
        """Implements a circuit that returns 1
        if the chain moves in the direction z-."""
        return (1-q_i) * q_j * (1-q_k)

    def circuit_000(self, q_i, q_j, q_k):
        """Implements a circuit that checks the
        nonsensical string 000."""
        return (1-q_i) * (1-q_j) * (1-q_k)

    def circuit_011(self, q_i, q_j, q_k):
        """Implements a circuit that checks the
        nonsensical string 000."""
        return (1-q_i) * q_j * q_k
    
    def x_position(self, n):
        """Computes the x coordinate of the nth residue."""

        if n > self.naas: 
            raise ValueError('n greater than number of residues') 
        return sum([
            self.circuit_xp(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1),
                            self.get(self.r_pointer(i+1)+2)) - \
            self.circuit_xn(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1),
                            self.get(self.r_pointer(i+1)+2))
        for i in range(n)]) 
    
    def y_position(self, n):
        """Computes the x coordinate of the nth residue."""

        if n > self.naas: 
            raise ValueError('n greater than number of residues') 
        return sum([
            self.circuit_yp(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1),
                            self.get(self.r_pointer(i+1)+2)) - \
            self.circuit_yn(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1),
                            self.get(self.r_pointer(i+1)+2))
        for i in range(n)]) 
    
    def z_position(self, n):
        """Computes the x coordinate of the nth residue."""

        if n > self.naas: 
            raise ValueError('n greater than number of residues') 
        return sum([
            self.circuit_zp(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1),
                            self.get(self.r_pointer(i+1)+2)) - \
            self.circuit_zn(self.get(self.r_pointer(i+1)), 
                            self.get(self.r_pointer(i+1)+1),
                            self.get(self.r_pointer(i+1)+2))
        for i in range(n)]) 
    
    def g(self, i, j):
        """Computes the distance between residues i and j."""

        return (self.x_position(i) - self.x_position(j))**2 \
             + (self.y_position(i) - self.y_position(j))**2 \
             + (self.z_position(i) - self.z_position(j))**2
    
    def mu(self, i, j):
        """Computes \mu_{ij}."""
    
        if i == j:
            return 0
        elif abs(i-j) < 3:
            return 0
        else:
            return 2 * int(math.ceil(math.log2(abs(i-j)))) \
                     * ((1+i-j) % 2)
    
    def alpha(self, i, j):
        """Computes \alpha_{ij}."""
        return sum([
            2**k * self.get(self.o_pointer(i, j) + k)
        for k in range(self.mu(i,j))])

    def redun_term(self):
        """Implements the term that penalises meaningless 
        residue bitstrings 000 and 011."""

        return sum([
            self.circuit_000(self.get(self.r_pointer(k)),
                             self.get(self.r_pointer(k)+1),
                             self.get(self.r_pointer(k)+2)) + \
            self.circuit_011(self.get(self.r_pointer(k)),
                             self.get(self.r_pointer(k)+1),
                             self.get(self.r_pointer(k)+2))
        for k in range(self.naas)])

    def back_term(self):
        """Ensures that the chain does not go
        back on itself."""

        return sum([
            self.circuit_xp(self.get(self.r_pointer(i)),
                            self.get(self.r_pointer(i)+1),
                            self.get(self.r_pointer(i)+2)) *
            self.circuit_xn(self.get(self.r_pointer(i+1)),
                            self.get(self.r_pointer(i+1)+1),
                            self.get(self.r_pointer(i+1)+2)) + \
            self.circuit_xn(self.get(self.r_pointer(i)),
                            self.get(self.r_pointer(i)+1),
                            self.get(self.r_pointer(i)+2)) *
            self.circuit_xp(self.get(self.r_pointer(i+1)),
                            self.get(self.r_pointer(i+1)+1),
                            self.get(self.r_pointer(i+1)+2)) + \
            self.circuit_yp(self.get(self.r_pointer(i)),
                            self.get(self.r_pointer(i)+1),
                            self.get(self.r_pointer(i)+2)) *
            self.circuit_yn(self.get(self.r_pointer(i+1)),
                            self.get(self.r_pointer(i+1)+1),
                            self.get(self.r_pointer(i+1)+2)) + \
            self.circuit_yn(self.get(self.r_pointer(i)),
                            self.get(self.r_pointer(i)+1),
                            self.get(self.r_pointer(i)+2)) *
            self.circuit_yp(self.get(self.r_pointer(i+1)),
                            self.get(self.r_pointer(i+1)+1),
                            self.get(self.r_pointer(i+1)+2)) + \
            self.circuit_zp(self.get(self.r_pointer(i)),
                            self.get(self.r_pointer(i)+1),
                            self.get(self.r_pointer(i)+2)) *
            self.circuit_zn(self.get(self.r_pointer(i+1)),
                            self.get(self.r_pointer(i+1)+1),
                            self.get(self.r_pointer(i+1)+2)) + \
            self.circuit_zn(self.get(self.r_pointer(i)),
                            self.get(self.r_pointer(i)+1),
                            self.get(self.r_pointer(i)+2)) *
            self.circuit_zp(self.get(self.r_pointer(i+1)),
                            self.get(self.r_pointer(i+1)+1),
                            self.get(self.r_pointer(i+1)+2))
        for i in range(self.naas-1)])

    def steric_term(self):
        """Ensures that the chain does not overlap."""
    
        term = sp.numbers.Integer(0)
        for i in range(self.naas-4):
            for j in range(i+4, self.naas):
                if (1+i-j) % 2:
                    term += (2**self.mu(i, j) -self.g(i, j) \
                         -  self.alpha(i, j))**2
        return term

    def interaction_term_ij(self, i, j):
        return -1 * self.get(self.i_pointer(i, j)) \
                           * self.int_mat[i, j] \
                           * ( 2 - self.g(i, j) )
    
    def interaction_term(self):
        """Computes contacts between residues."""
    
        term = sp.numbers.Integer(0)
        for i in range(self.naas-3):
            for j in range(i+3, self.naas):
                if self.int_mat[i, j] == 0: continue
                term -= self.get(self.i_pointer(i, j)) \
                      * self.int_mat[i, j] \
                      * ( 2 - self.g(i, j) )
        return term

