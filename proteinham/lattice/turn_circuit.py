import math
import numpy as np
import sympy as sp
import symengine as se
from abc import *
from tqdm import tqdm, trange
from copy import deepcopy
from itertools import chain
from functools import reduce

from .qlogic import *
from proteinham.core.hamiltonian import Hamiltonian


class CommonTurnCircuitHamiltonian(Hamiltonian):

    is_TurnCircuit = True

    def __init__(self, pepstring, ss_fmat='babej'):
        """Encapsulates the expression and methods of
        a protein hamiltonian of the "turn circuit encoding" 
        form, described by Babbush et al., 2012."""
  
        self._proc_input(pepstring) 
        self.ss_fmat = ss_fmat
        self.n_bits = self.dim * (self.naas-1)
        self._sum_strings = dict()
        self._create_bitreg()

    @property
    def encoding(self):
        return 'turn_circuit'

    def build_exp(self): 
        self.expr      = (self.naas+1) * self.back_term()
        if self.dim == 3:
            self.expr += (self.naas+1)**2 * self.redun_term()
        self.expr     += (self.naas+1) * self.steric_term()
        self.expr     += self.interaction_term()
        #self.expr      = se.expand(self.expr)
        self.n_terms   = len(self.expr.args)

    def get(self, k):
        """Access the kth bit of the hamiltonian."""
        return self.bit_list[k]
 
    def half_adder(self, q_i, q_j):
        """Applies a half-adder."""
        return qand([q_i, q_j]), qxor(q_i, q_j)

    @property
    @abstractmethod
    def dim(self):
        pass


class TurnCircuitHamiltonian2D(CommonTurnCircuitHamiltonian):

    is_2D = True

    @property
    def dim(self):
        return 2

    def pointer(self, i):
        """Points to the start of the string describing
        the ith turn."""
        return 2*i
    
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

    def sum_string(self, i, j, k):
        """Computes the sum string."""
        if i > j:
            raise ValueError("i > j")

        if (i, j, k) in self._sum_strings.keys():
            return self._sum_strings[(i, j, k)]
    
        if k == 'x+':
            sum_string = [self.circuit_xp(self.get(self.pointer(t)),
                                          self.get(self.pointer(t)+1))
                          for t in range(i, j)]
        elif k == 'x-':
            sum_string = [self.circuit_xn(self.get(self.pointer(t)),
                                          self.get(self.pointer(t)+1))
                          for t in range(i, j)]
    
        elif k == 'y+':
            sum_string = [self.circuit_yp(self.get(self.pointer(t)),
                                          self.get(self.pointer(t)+1))
                          for t in range(i, j)]
    
        elif k == 'y-':
            sum_string = [self.circuit_yn(self.get(self.pointer(t)),
                                          self.get(self.pointer(t)+1))
                          for t in range(i, j)]
    
        else:
            raise ValueError('k was {:s}'.format(k))
   
        n_layers = j-i-1
        counter = np.zeros(n_layers) # lazy way to keep track of half-adders
        sum_string = list(reversed(sum_string))
        for t in chain(range(n_layers),
                       reversed(range(n_layers-1))):
    
            if t % 2 == 0:
                iterator = range(0, t+1, 2) if t > 0 else [0]
            else:
                iterator = range(1, t+1, 2) if t > 1 else [1]

            for h in iterator:
    
                if self.ss_fmat == 'babej':
                    if counter[h] > math.log2(j-i):
                        continue
                    else:
                        counter[h] += 1

                a, b = self.half_adder(sum_string[h],
                                       sum_string[h+1])
                sum_string[h]   = a
                sum_string[h+1] = b

        maximum = int(math.ceil(math.log2(j-i)))
        sum_string = list(reversed(sum_string))
        self._sum_strings[(i, j, k)] = [sp.expand(sum_string[x]) for x in range(maximum)]
        return self._sum_strings[(i, j, k)]

    def back_term(self):
        """Ensures that the chain does not go
        back on itself."""

        return sum([
            self.circuit_xp(self.get(self.pointer(i)),
                            self.get(self.pointer(i)+1)) *
            self.circuit_xn(self.get(self.pointer(i+1)),
                            self.get(self.pointer(i+1)+1)) + \
            self.circuit_xn(self.get(self.pointer(i)),
                            self.get(self.pointer(i)+1)) *
            self.circuit_xp(self.get(self.pointer(i+1)),
                            self.get(self.pointer(i+1)+1)) + \
            self.circuit_yp(self.get(self.pointer(i)),
                            self.get(self.pointer(i)+1)) *
            self.circuit_yn(self.get(self.pointer(i+1)),
                            self.get(self.pointer(i+1)+1)) + \
            self.circuit_yn(self.get(self.pointer(i)),
                            self.get(self.pointer(i)+1)) *
            self.circuit_yp(self.get(self.pointer(i+1)),
                            self.get(self.pointer(i+1)+1))
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
    
        return prefactor * \
        ( qxor(sumstring['x+'][0],
               sumstring['x-'][0]) \
        * qand([
            qxnor(sumstring['x+'][r],
                  sumstring['x-'][r])
        for r in range(1, maximum)]) \
        + sum([
            qxor(sumstring['x+'][p-2],
                 sumstring['x+'][p-1]) \
          * qand([
                qxnor(sumstring['x+'][r-1],
                      sumstring['x+'][r])
            for r in range(1, p-1)]) \
          * qand([
                qxor(sumstring['x+'][r-1],
                     sumstring['x-'][r-1])
            for r in range(1, p+1)])  \
          * qand([
                qxnor(sumstring['x+'][r-1],
                      sumstring['x-'][r-1])
            for r in range(p+1, maximum+1)])
        for p in range(2, maximum+1)]))
    
    def a_y(self, i, j):
    
        sumstring = {
            'x+': self.sum_string(i, j, 'x+'),
            'x-': self.sum_string(i, j, 'x-'),
            'y+': self.sum_string(i, j, 'y+'),
            'y-': self.sum_string(i, j, 'y-')
        }
    
        maximum = int(math.ceil(math.log2(abs(i-j)))) if i-j != 0 else 0
    
        if maximum == 0: return 0
    
        prefactor = qand([
            qxnor(sumstring['x+'][r],
                  sumstring['x-'][r])
        for r in range(maximum)])
    
        return prefactor *\
        ( qxor(sumstring['y+'][0],
               sumstring['y-'][0])  \
        * qand([
            qxnor(sumstring['y+'][r],
                  sumstring['y-'][r])
        for r in range(1, maximum)]) \
        + sum([
            qxor(sumstring['y+'][p-2],
                 sumstring['y+'][p-1]) \
          * qand([
                qxnor(sumstring['y+'][r-1],
                      sumstring['y+'][r])
            for r in range(1, p-1)]) \
          * qand([
                qxor(sumstring['y+'][r-1],
                     sumstring['y-'][r-1])
            for r in range(1, p+1)])  \
          * qand([
                qxnor(sumstring['y+'][r-1],
                      sumstring['y-'][r-1])
            for r in range(p+1, maximum+1)])
        for p in range(2, maximum+1)]))

    def interaction_term_ij(self, i, j):
        return -1 * self.int_mat[i, j] * (self.a_x(i, j) + \
                                          self.a_y(i, j))
    
    def interaction_term(self):
        """Computes contacts between residues."""
    
        expr = sp.numbers.Integer(0)
        for i in range(self.naas-3):
            for j in range(1, math.ceil((self.naas-i-1)/2)):
    
                if self.int_mat[i, 1+i+2*j] == 0: continue
    
                expr += self.interaction_term_ij(i, 1+i+2*j)
    
        return expr


class TurnCircuitHamiltonian3D(CommonTurnCircuitHamiltonian):

    is_3D = True

    @property
    def dim(self):
        return 3

    def pointer(self, i):
        """Points to the start of the string describing
        the ith turn."""
        return 3*i

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

    def sum_string(self, i, j, k):
        """Computes the sum string."""
        if i > j:
            raise ValueError("i > j")

        if (i, j, k) in self._sum_strings.keys():
            return self._sum_strings[(i, j, k)]
    
        if k == 'x+':
            sum_string = [self.circuit_xp(self.get(self.pointer(t)),
                                          self.get(self.pointer(t)+1),
                                          self.get(self.pointer(t)+2))
                          for t in range(i, j)]
        elif k == 'x-':
            sum_string = [self.circuit_xn(self.get(self.pointer(t)),
                                          self.get(self.pointer(t)+1),
                                          self.get(self.pointer(t)+2))
                          for t in range(i, j)]
    
        elif k == 'y+':
            sum_string = [self.circuit_yp(self.get(self.pointer(t)),
                                          self.get(self.pointer(t)+1),
                                          self.get(self.pointer(t)+2))
                          for t in range(i, j)]
    
        elif k == 'y-':
            sum_string = [self.circuit_yn(self.get(self.pointer(t)),
                                          self.get(self.pointer(t)+1),
                                          self.get(self.pointer(t)+2))
                          for t in range(i, j)]
    
        elif k == 'z+':
            sum_string = [self.circuit_zp(self.get(self.pointer(t)),
                                          self.get(self.pointer(t)+1),
                                          self.get(self.pointer(t)+2))
                          for t in range(i, j)]

        elif k == 'z-':
            sum_string = [self.circuit_zn(self.get(self.pointer(t)),
                                          self.get(self.pointer(t)+1),
                                          self.get(self.pointer(t)+2))
                          for t in range(i, j)]
        else:
            raise ValueError('k was {:s}'.format(k))
   
        n_layers = j-i-1
        counter = np.zeros(n_layers) # lazy way to keep track of half-adders
        sum_string = list(reversed(sum_string))
        for t in chain(range(n_layers),
                       reversed(range(n_layers-1))):
    
            if t % 2 == 0:
                iterator = range(0, t+1, 2) if t > 0 else [0]
            else:
                iterator = range(1, t+1, 2) if t > 1 else [1]

            for h in iterator:
    
                if self.ss_fmat == 'babej':
                    if counter[h] > math.log2(j-i):
                        continue
                    else:
                        counter[h] += 1

                a, b = self.half_adder(sum_string[h],
                                       sum_string[h+1])
                sum_string[h]   = a
                sum_string[h+1] = b

        maximum = int(math.ceil(math.log2(j-i)))
        sum_string = list(reversed(sum_string))
        self._sum_strings[(i, j, k)] = [sp.expand(sum_string[x]) for x in range(maximum)]
        return self._sum_strings[(i, j, k)]

    def redun_term(self):
        """Implements the term that penalises meaningless 
        residue bitstrings 000 and 011."""

        return sum([
            self.circuit_000(self.get(self.pointer(k)),
                             self.get(self.pointer(k)+1),
                             self.get(self.pointer(k)+2)) + \
            self.circuit_011(self.get(self.pointer(k)),
                             self.get(self.pointer(k)+1),
                             self.get(self.pointer(k)+2))
        for k in range(self.naas-1)])

    def back_term(self):
        """Ensures that the chain does not go
        back on itself."""

        return sum([
            self.circuit_xp(self.get(self.pointer(i)),
                            self.get(self.pointer(i)+1),
                            self.get(self.pointer(i)+2)) *
            self.circuit_xn(self.get(self.pointer(i+1)),
                            self.get(self.pointer(i+1)+1),
                            self.get(self.pointer(i+1)+2)) + \
            self.circuit_xn(self.get(self.pointer(i)),
                            self.get(self.pointer(i)+1),
                            self.get(self.pointer(i)+2)) *
            self.circuit_xp(self.get(self.pointer(i+1)),
                            self.get(self.pointer(i+1)+1),
                            self.get(self.pointer(i+1)+2)) + \
            self.circuit_yp(self.get(self.pointer(i)),
                            self.get(self.pointer(i)+1),
                            self.get(self.pointer(i)+2)) *
            self.circuit_yn(self.get(self.pointer(i+1)),
                            self.get(self.pointer(i+1)+1),
                            self.get(self.pointer(i+1)+2)) + \
            self.circuit_yn(self.get(self.pointer(i)),
                            self.get(self.pointer(i)+1),
                            self.get(self.pointer(i)+2)) *
            self.circuit_yp(self.get(self.pointer(i+1)),
                            self.get(self.pointer(i+1)+1),
                            self.get(self.pointer(i+1)+2)) + \
            self.circuit_zp(self.get(self.pointer(i)),
                            self.get(self.pointer(i)+1),
                            self.get(self.pointer(i)+2)) *
            self.circuit_zn(self.get(self.pointer(i+1)),
                            self.get(self.pointer(i+1)+1),
                            self.get(self.pointer(i+1)+2)) + \
            self.circuit_zn(self.get(self.pointer(i)),
                            self.get(self.pointer(i)+1),
                            self.get(self.pointer(i)+2)) *
            self.circuit_zp(self.get(self.pointer(i+1)),
                            self.get(self.pointer(i+1)+1),
                            self.get(self.pointer(i+1)+2))
        for i in range(self.naas-2)])

    def overlap(self, i, j):
        """Computes the overlap term for residues i and j."""

        maximum = int(math.ceil(math.log2(abs(i-j)))) if i-j != 0 else 0
        if (j-i) % 2 != 0 or maximum < 2: return sp.numbers.Integer(0)

        sumstring = {
            'x+': self.sum_string(i, j, 'x+'),
            'x-': self.sum_string(i, j, 'x-'),
            'y+': self.sum_string(i, j, 'y+'),
            'y-': self.sum_string(i, j, 'y-'),
            'z+': self.sum_string(i, j, 'z+'),
            'z-': self.sum_string(i, j, 'z-'),
        }

        return qand(
            [qxnor(sumstring['x+'][r],
                   sumstring['x-'][r])
            for r in range(maximum)] + \
            [qxnor(sumstring['y+'][r],
                   sumstring['y-'][r])
            for r in range(maximum)] + \
            [qxnor(sumstring['z+'][r],
                   sumstring['z-'][r])
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
            'y-': self.sum_string(i, j, 'y-'),
            'z+': self.sum_string(i, j, 'z+'),
            'z-': self.sum_string(i, j, 'z-')
        }

        maximum = int(math.ceil(math.log2(abs(i-j)))) if i-j !=0 else 0
        if maximum == 0: return 0

        prefactor = qand([
            qand([
                qxnor(sumstring['%s+' % k][r],
                      sumstring['%s-' % k][r])
            for r in range(maximum)])
        for k in ['y', 'z']])

        return prefactor * \
        ( qxor(sumstring['x+'][0],
               sumstring['x-'][0]) \
        * qand([
            qxnor(sumstring['x+'][r],
                  sumstring['x-'][r])
        for r in range(1, maximum)]) \
        + sum([
            qxor(sumstring['x+'][p-2],
                 sumstring['x+'][p-1]) \
          * qand([
                qxnor(sumstring['x+'][r-1],
                      sumstring['x+'][r])
            for r in range(1, p-1)]) \
          * qand([
                qxor(sumstring['x+'][r-1],
                     sumstring['x-'][r-1])
            for r in range(1, p+1)])  \
          * qand([
                qxnor(sumstring['x+'][r-1],
                      sumstring['x-'][r-1])
            for r in range(p+1, maximum+1)])
        for p in range(2, maximum+1)]))

    def a_y(self, i, j):

        sumstring = {
            'x+': self.sum_string(i, j, 'x+'),
            'x-': self.sum_string(i, j, 'x-'),
            'y+': self.sum_string(i, j, 'y+'),
            'y-': self.sum_string(i, j, 'y-'),
            'z+': self.sum_string(i, j, 'z+'),
            'z-': self.sum_string(i, j, 'z-')
        }

        maximum = int(math.ceil(math.log2(abs(i-j)))) if i-j !=0 else 0
        if maximum == 0: return 0

        prefactor = qand([
            qand([
                qxnor(sumstring['%s+' % k][r],
                      sumstring['%s-' % k][r])
            for r in range(maximum)])
        for k in ['x', 'z']])

        return prefactor * \
        ( qxor(sumstring['y+'][0],
               sumstring['y-'][0]) \
        * qand([
            qxnor(sumstring['y+'][r],
                  sumstring['y-'][r])
        for r in range(1, maximum)]) \
        + sum([
            qxor(sumstring['y+'][p-2],
                 sumstring['y+'][p-1]) \
          * qand([
                qxnor(sumstring['y+'][r-1],
                      sumstring['y+'][r])
            for r in range(1, p-1)]) \
          * qand([
                qxor(sumstring['y+'][r-1],
                     sumstring['y-'][r-1])
            for r in range(1, p+1)])  \
          * qand([
                qxnor(sumstring['y+'][r-1],
                      sumstring['y-'][r-1])
            for r in range(p+1, maximum+1)])
        for p in range(2, maximum+1)]))

    def a_z(self, i, j):

        sumstring = {
            'x+': self.sum_string(i, j, 'x+'),
            'x-': self.sum_string(i, j, 'x-'),
            'y+': self.sum_string(i, j, 'y+'),
            'y-': self.sum_string(i, j, 'y-'),
            'z+': self.sum_string(i, j, 'z+'),
            'z-': self.sum_string(i, j, 'z-')
        }

        maximum = int(math.ceil(math.log2(abs(i-j)))) if i-j !=0 else 0
        if maximum == 0: return 0

        prefactor = qand([
            qand([
                qxnor(sumstring['%s+' % k][r],
                      sumstring['%s-' % k][r])
            for r in range(maximum)])
        for k in ['x', 'y']])

        return prefactor * \
        ( qxor(sumstring['z+'][0],
               sumstring['z-'][0]) \
        * qand([
            qxnor(sumstring['z+'][r],
                  sumstring['z-'][r])
        for r in range(1, maximum)]) \
        + sum([
            qxor(sumstring['z+'][p-2],  
                 sumstring['z+'][p-1]) \
          * qand([
                qxnor(sumstring['z+'][r-1],
                      sumstring['z+'][r])
            for r in range(1, p-1)]) \
          * qand([
                qxor(sumstring['z+'][r-1],
                     sumstring['z-'][r-1])
            for r in range(1, p+1)])  \
          * qand([
                qxnor(sumstring['z+'][r-1],
                      sumstring['z-'][r-1]) 
            for r in range(p+1, maximum+1)])
        for p in range(2, maximum+1)]))

    def interaction_term_ij(self, i, j):
        return -1* self.int_mat[i, j] * (self.a_x(i, j) + \
                                         self.a_y(i, j) + \
                                         self.a_z(i, j))

    def interaction_term(self):
        """Computes contacts between residues."""

        expr = sp.numbers.Integer(0)
        for i in range(self.naas-3):
            for j in range(1, math.ceil((self.naas-i-1)/2)):

                if self.int_mat[i, 1+i+2*j] == 0: continue

                expr += interaction_term_ij(i, 1+i+2*j)

        return expr
