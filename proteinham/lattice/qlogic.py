import numpy as np
import sympy as sp
from functools import reduce


def qand(q_list):
    """Applies an AND operation between
    all bits in a list."""

    assert np.all([type(x) is sp.Symbol or sp.add.Add or sp.mul.Mul
                   for x in q_list])
    if len(q_list) == 0:
        return 1
    else:
        return reduce(lambda x, y: x*y, q_list)

def qxnor(q_i, q_j):
    """Applies an XNOR operation between
    bits i and j."""

    assert type(q_i) is sp.Symbol or sp.add.Add or sp.mul.Mul
    assert type(q_j) is sp.Symbol or sp.add.Add or sp.mul.Mul
    return 1 -q_i -q_j + 2*q_i*q_j

def qnot(q_i):
    """Applies a NOT operation to bit i."""

    assert type(q_i) is sp.Symbol or sp.add.Add or sp.mul.Mul
    return 1-q_i

def qor(q_i, q_j):
    """Applies an OR operation between
    bits i and j."""

    assert type(q_i) is sp.Symbol or sp.add.Add or sp.mul.Mul
    assert type(q_j) is sp.Symbol or sp.add.Add or sp.mul.Mul
    return q_i +q_j -q_i*q_j

def qxor(q_i, q_j):
    """Applies a XOR operation between
    bits i and j."""

    assert type(q_i) is sp.Symbol or sp.add.Add or sp.mul.Mul
    assert type(q_j) is sp.Symbol or sp.add.Add or sp.mul.Mul
    return q_i + q_j - 2*q_i*q_j

