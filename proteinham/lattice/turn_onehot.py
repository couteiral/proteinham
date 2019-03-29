import math
import numpy as np
import sympy as sp
from qlogic import *
from tqdm import tqdm, trange
from copy import deepcopy
from itertools import chain
from functools import reduce
from int_matrix import int_matrix


class TurnOneHotHamiltonian2D(object):

    is_TurnOneHot = True
    is_2D         = True

    def __init__(self, pepstring):
        """Encapsulates the expression and methods of
        a protein hamiltonian of the "turn one-hot encoding" 
        form, described by Fingerhuth et al., 2018.

        We employ the convention:

            RIGHT -> 1000
            UP    -> 0100
            LEFT  -> 0010
            DOWN  -> 0001
        """
   
        self.naas     = len(pepstring)
        self.dim      = 2
        self.n_bits   = 4*self.naas-4
        self.int_mat  = int_matrix(pepstring)

        self.bit_list = [
            sp.Symbol('q_{:d}'.format(i+1), idempotent=True)
            for i in range(self.n_bits)
        ]
    
        self.expr     = self.one_term()
        self.expr    += self.back_term()
        self.expr    += self.steric_term()
        self.expr    += self.interaction_term()
        self.expr     = sp.expand(self.expr)

    def get(self, k):
        """Access the kth bit of the hamiltonian."""
        return self.bit_list[k]

    def get_turn(self, k):
        """Access the bits encoding the kth turn."""
        return self.bit_list[self.pointer(k):self.pointer(k)+4]
 
    def pointer(self, i):
        """Points to the start of the string describing
        the ith turn."""
        return 2*i
    
    def circuit_xp(self, i):
        """Implements a circuit that returns 1
        if the chain moves in the direction x+."""
        q_i, q_j, q_k, q_l = self.get_turn(i)
        return q_i*(1-q_j)*(1-q_k)*(1-q_l)
    
    def circuit_xn(self, i):
        """Implements a circuit that returns 1
        if the chain moves in the direction x-."""
        q_i, q_j, q_k, q_l = self.get_turn(i)
        return (1-q_i)*(1-q_j)*q_k*(1-q_l)
    
    def circuit_yp(self, i):
        """Implements a circuit that returns 1
        if the chain moves in the direction y+."""
        q_i, q_j, q_k, q_l = self.get_turn(i)
        return (1-q_i)*q_j*(1-q_k)*(1-q_l)
    
    def circuit_yn(self, i):
        """Implements a circuit that returns 1
        if the chain moves in the direction y-."""
        q_i, q_j, q_k, q_l = self.get_turn(i)
        return (1-q_i)*(1-q_j)*(1-q_j)*q_l
    
    def half_adder(self, q_i, q_j):
        """Applies a half-adder."""
        return qand([q_i, q_j]), qxor(q_i, q_j)
    
    def sum_string(self, i, j, k):
        """Computes the sum string."""
    
        if i > j:
            raise ValueError("i > j")
        ip = i
        jp = j
    
        if k == 'x+':
            sum_string = [self.circuit_xp(t)
                          for t in range(ip, jp)]
        elif k == 'x-':
            sum_string = [self.circuit_xn(t)
                          for t in range(ip, jp)]
    
        elif k == 'y+':
            sum_string = [self.circuit_yp(t)
                          for t in range(ip, jp)]
    
        elif k == 'y-':
            sum_string = [self.circuit_yn(t)
                          for t in range(ip, jp)]
    
        else:
            raise ValueError('k was {:s}'.format(k))
    
        n_layers = jp-ip-1
        sum_string = list(reversed(sum_string))
    
        for t in chain(range(n_layers),
                       reversed(range(n_layers-1))):
    
            if t % 2 == 0:
                iterator = range(0, t+1, 2) if t > 0 else [0]
            else:
                iterator = range(1, t+1, 2) if t > 1 else [1]
            for h in iterator:
    
                a, b = self.half_adder(sum_string[h],
                                       sum_string[h+1])
                sum_string[h]   = a
                sum_string[h+1] = b
    
        return [sp.expand(x) for x in reversed(sum_string)]

    def one_term(self):
        """Implements the term tthat avoids multiple
        bits turned on for the same residue."""

        return sum([
            sum([
                sum([
                    self.get(self.pointer(k)+i) *
                    self.get(self.pointer(k)+j)
                for j in range(i+1, 4)])
            for i in range(3)])
        for k in range(1, self.naas)])

    def back_term(self):
        """Ensures that the chain does not go
        back on itself."""

        return sum([
            self.circuit_xp(i) * \
            self.circuit_xn(i+1) + \
            self.circuit_xn(i) * \
            self.circuit_xp(i+1) + \
            self.circuit_yp(i) * \
            self.circuit_yn(i+1) + \
            self.circuit_yn(i) * \
            self.circuit_yp(i+1)
        for i in range(self.naas-2)])
    
    def overlap(self, i, j):
        """Computes the overlap term for residues i and j."""
    
        maximum = int(math.ceil(math.log2(abs(i-j)))) if i-j != 0 else 0
        if (j-i) % 2 != 0 or maximum < 2: return sp.numbers.Integer(0)
    
        sumstring = {
            'x+': self.sum_string(i, j, 'x+'),
            'x-': self.sum_string(i, j, 'x-'),
            'y+': self.sum_string(i, j, 'y+'),
            'y-': self.sum_string(i, j, 'y-')
        }
    
        return qand(
            [qxnor(sumstring['x+'][r],
                   sumstring['x-'][r])
            for r in range(maximum)] + \
            [qxnor(sumstring['y+'][r],
                   sumstring['y-'][r])
            for r in range(maximum)]
        )
            
    def steric_term(self):
        """Ensures that the chain does not overlap."""
    
        return sum([
            sum([
                self.overlap(i, j)
            for j in range(i+1, self.naas)])
        for i in range(self.naas)])
    
    def a_x(self, i, j):
    
        sumstring = {
            'x+': self.sum_string(i, j, 'x+'),
            'x-': self.sum_string(i, j, 'x-'),
            'y+': self.sum_string(i, j, 'y+'),
            'y-': self.sum_string(i, j, 'y-')
        }
    
        maximum = int(math.ceil(math.log2(abs(i-j)))) if i-j !=0 else 0
        if maximum == 0: return 0
    
        prefactor = qand([
            qxnor(sumstring['y+'][r],
                  sumstring['y-'][r])
        for r in range(maximum)])
    
        return prefactor * (\
        qxor(sumstring['x+'][0],
             sumstring['x-'][0]) * \
        qand([
            qxnor(sumstring['x+'][r],
                  sumstring['x-'][r])
        for r in range(1, maximum)]) + \
        sumstring['x+'][0]*sumstring['x-'][0] * \
        sum([
            qxor(sumstring['x+'][p-1],
                 sumstring['x+'][p]) * \
            qand([
                qxnor(sumstring['x+'][r],
                      sumstring['x+'][r+1])
            for r in range(p-2)]) * \
            qand([
                qxor(sumstring['x+'][r],
                     sumstring['x-'][r])
            for r in range(p)]) * \
            qand([
                qxnor(sumstring['x+'][r],
                      sumstring['x-'][r])
            for r in range(p+1, maximum)])
        for p in range(1, maximum)]))
    
    def a_y(self, i, j):
    
        sumstring = {
            'x+': self.sum_string(i, j, 'x+'),
            'x-': self.sum_string(i, j, 'x-'),
            'y+': self.sum_string(i, j, 'y+'),
            'y-': self.sum_string(i, j, 'y-')
        }
    
        maximum = int(math.ceil(math.log2(abs(i-j)))) if i-j !=0 else 0
    
        if maximum == 0: return 0
    
        prefactor = qand([
            qxnor(sumstring['x+'][r],
                  sumstring['x-'][r])
        for r in range(maximum)])
    
        return prefactor * (\
        qxor(sumstring['y+'][0],
             sumstring['y-'][0]) * \
        qand([
            qxnor(sumstring['y+'][r],
                  sumstring['y-'][r])
        for r in range(1, maximum)]) + \
        sumstring['y+'][0]*sumstring['y-'][0] * \
        sum([
            qxor(sumstring['y+'][p-1],
                 sumstring['y+'][p]) * \
            qand([
                qxnor(sumstring['y+'][r],
                      sumstring['y+'][r+1])
            for r in range(p-2)]) * \
            qand([
                qxor(sumstring['y+'][r],
                     sumstring['y-'][r])
            for r in range(p)]) * \
            qand([
                qxnor(sumstring['y+'][r],
                      sumstring['y-'][r])
            for r in range(p+1, maximum)])
        for p in range(1, maximum)]))
    
    def interaction_term(self):
        """Computes contacts between residues."""
    
        expr = sp.numbers.Integer(0)
        for i in range(self.naas-3):
            for j in range(1, math.ceil((self.naas-i-1)/2)):
    
                if self.int_mat[i, 1+i+2*j] == 0: continue
    
                expr += self.int_mat[i, 1+i+2*j] * (self.a_x(i, 1+i+2*j) + \
                                                    self.a_y(i, 1+i+2*j))
    
        return expr

