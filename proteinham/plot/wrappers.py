import math
import numpy as np
import warnings
from diamond_utils import n_k, rhombus


def process_bead(bitstring, N):

    assert type(N) is int
    assert type(bitstring) is str or list or np.array
    if type(bitstring) is list or np.array:
        bitstr = ''.join(bitstring)
    else:
        bitstr = bitstring
    bitstr = bitstr[::-1]

    n_bits = int( math.ceil(math.log(N)) )
    assert 2 * n_bits * N <= len(bitstring)

    peptide = np.zeros((N, 2))
    for i in range(N):
        aastr = bitstr[2*n_bits*i:2*n_bits*(i+1)]

        peptide[i, 0] = int(aastr[n_bits:], base=2)
        peptide[i, 1] = int(aastr[:n_bits], base=2)

    return peptide

def process_turn(bitstring, N):

    assert type(N) is int
    assert type(bitstring) is str or list or np.array
    if type(bitstring) is list or np.array:
        bitstr = ''.join(bitstring)
    else:
        bitstr = bitstring
    assert (N-1) * 2 <= len(bitstring)

    peptide = np.zeros((N, 2))
    for i in range(N-1):

        aastr = bitstr[2*i:2*(i+1)]
        x, y = turn_code(aastr)
        peptide[i+1, 0] = peptide[i, 0] + x
        peptide[i+1, 1] = peptide[i, 1] + y

    return peptide
        
def process_diamond(bitstring, N):

    assert type(N) is int
    assert type(bitstring) is str or list or np.array
    if type(bitstring) is list or np.array:
        bitstr = ''.join(bitstring)
    else:
        bitstr = bitstring

    pointer = lambda x: sum([n_k(t) for t in range(x)])
    assert pointer(N) <= len(bitstring)

    peptide = np.zeros((N, 2))
    for i in range(N-1):

        aastr = bitstr[pointer(i+1):pointer(i+1)+n_k(i+1)]
        if aastr.count('1') != 1:
            warnings.warn('String ' + bitstr + ' doesn\'t have only one active bit for ' +\
                          'the {:d}th residue'.format(i), RuntimeWarning)

        ind = aastr.index('1')
        peptide[i+1, :] = rhombus(i+1)[ind]

    return peptide

def turn_code(string):

    if   string == '00':
        return ( 0, -1 )
    elif string == '01':
        return ( 1,  0 )
    elif string == '10':
        return (-1,  0 )
    elif string == '11':
        return ( 0,  1 )
    else:
        raise ValueError('Wrong turn code')
   
