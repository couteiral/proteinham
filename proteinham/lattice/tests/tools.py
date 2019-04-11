import sympy as sp
from copy import deepcopy


def test_bitstring(expr, bitstring, n_bits):

    assert issubclass(type(expr), sp.Expr)
    assert type(bitstring) is str
    assert type(n_bits) is int and n_bits > 0
    assert len(bitstring) == n_bits
    assert set(bitstring).issubset({'1', '0'})

    value = deepcopy(expr)
    bit_list = [sp.Symbol('q_%d' % (i+1)) for i in range(n_bits)]
    for ind, val in enumerate(bitstring):
        value = value.subs(bit_list[ind],
                           int(val))

    return value
