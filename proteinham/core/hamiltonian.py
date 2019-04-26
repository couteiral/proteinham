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

        self.circuit = sp.expand(circuit)

    def test_bitstring(self, bitstring):
        assert type(bitstring) is str
        assert len(bitstring) == self.n_bits
        assert set(bitstring).issubset({'1', '0'})

        value = deepcopy(self.expr)
        for ind, val in enumerate(bitstring):
            value = value.subs(self.bit_list[ind],
                               int(val))

        return value

    def write_maxsat(self, file_name, fmat='onehot'):

        with open(file_name, 'w') as f:

            if fmat == 'cnf':
                f.writelines(self._process_cnf())
            elif fmat == 'onehot':
                f.writelines(self._process_onehot())
            else:
                raise ValueError('')

    def _process_onehot(self):

        template   = '% 06.2f' + ' %3d' * (self.n_bits) + '\n'
        file_lines = list()

        file_lines.extend([
        #    '%s\n'    % self.pepstring,
            '%d %d\n' % (self.n_bits, self.n_terms)
        ])

        for term in self.expr.args:

            if term.is_number:
                penalty = float(term)
                bitterm = [0 for _ in range(self.n_bits)]

            elif term.is_Symbol:
                penalty = 1.0
                bitterm = [0 if i != self._get_index(term) else i+1
                           for i in range(self.n_bits)]
            else:

                if term.args[0].is_number:
                    penalty = float( term.args[0] )
                    bitterm = self._get_bitterm(term.args[1:])
                else:
                    penalty = 1.0
                    bitterm = self._get_bitterm(term.args)

            file_lines.append(template % (penalty, *bitterm))

        return file_lines

    def _process_cnf(self):

        file_lines = list()
        file_lines.extend([
        #    '%s\n'    % self.pepstring,
            '%d %d\n' % (self.n_bits, self.n_terms)
        ])

        for term in self.expr.args:

            if term.is_number:
                continue

            elif term.is_Symbol:
                penalty = 1.0
                clauses = [i+1 for i in range(self.n_bits)
                           if i == self._get_index(term)]
                
            else:

                if term.args[0].is_number:
                    penalty = float( term.args[0] )
                    clauses = self._get_clauses(term.args[1:])
                else:
                    penalty = 1.0
                    clauses = self._get_clauses(term.args)
   
            if penalty > 0:
                file_lines.append(
                    ('% 06.2f' + ' %d' * len(clauses) + '\n')
                     % (penalty, *clauses)
                )
            elif penalty < 0:
                file_lines.append(
                    ('% 06.2f' + ' -%d' * len(clauses) + '\n')
                     % (-penalty, *clauses)
                )


        return file_lines
    
    def _get_bitterm(self, args):
        active_bits = [self._get_index(x) for x in args]
        return [0 if i not in active_bits else i+1 for i in range(self.n_bits)]

    def _get_clauses(self, args):
        return [self._get_index(x)+1 for x in args]

    def _get_index(self, sym):
        return int(str(sym).split('_')[-1])-1
