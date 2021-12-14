from proteinham.lattice import TurnCircuitHamiltonian2D


def test_HPPH_2D():

    ham = TurnCircuitHamiltonian2D('HPPH')
    ham.build_exp()
    bitstrings = {
        '111000':   -1.000000,
        '110100':   -1.000000,
        '111100':   -1.000000,
        '010010':   -1.000000,
        '011110':   -1.000000,
        '100001':   -1.000000,
        '101101':   -1.000000,
        '001011':   -1.000000,
        '000111':   -1.000000,
        '110000':    4.000000,
        '011000':    4.000000,
        '100100':    4.000000,
        '101100':    4.000000,
        '011100':    4.000000,
        '110010':    4.000000,
        '011010':    4.000000,
        '010110':    4.000000,
        '110110':    4.000000,
        '001110':    4.000000,
        '110001':    4.000000,
        '001001':    4.000000,
        '101001':    4.000000,
        '111001':    4.000000,
        '100101':    4.000000,
        '001101':    4.000000,
        '000011':    4.000000,
        '100011':    4.000000,
        '010011':    4.000000,
        '011011':    4.000000,
        '100111':    4.000000,
        '001111':    4.000000,
        '001100':    9.000000,
        '100110':    9.000000,
        '011001':    9.000000,
        '110011':    9.000000
    }

    # Minimum and maximum values are ok
    for bstring, energy in bitstrings.items():
        eigval = ham.test_bitstring(bstring)
        assert eigval == energy

def test_HPPHP_2D():

    ham = TurnCircuitHamiltonian2D('HPPHP')
    ham.build_exp()

    bitstrings = {
        '11100000':   -1.000000,
        '11010000':   -1.000000,
        '01001000':   -1.000000,
        '10000100':   -1.000000,
        '11100010':   -1.000000,
        '01001010':   -1.000000,
        '01111010':   -1.000000,
        '00101110':   -1.000000,
        '11010001':   -1.000000,
        '10000101':   -1.000000,
        '10110101':   -1.000000,
        '01111011':   -1.000000,
        '10110111':   -1.000000,
        '00101111':   -1.000000,
        '00011111':   -1.000000,
        '11000111':    5.000000,
        '10100111':    5.000000,
        '11100111':    5.000000,
        '10010111':    5.000000,
        '00110111':    5.000000,
        '10001111':    5.000000,
        '01001111':    5.000000,
        '01101111':    5.000000,
        '10011111':    5.000000,
        '00111111':    5.000000,
        '00000000':    6.000000,
        '10101100':    6.000000,
        '11101100':    6.000000,
        '01011100':    6.000000,
        '11011100':    6.000000,
        '10111100':    6.000000,
        '01111100':    6.000000,
        '11111100':    6.000000,
        '10101010':    6.000000,
        '00000110':    6.000000,
        '01000110':    6.000000,
        '00010110':    6.000000,
        '01010110':    6.000000,
        '11010110':    6.000000,
        '01110110':    6.000000,
        '11110110':    6.000000,
        '00001001':    6.000000,
        '10001001':    6.000000,
        '00101001':    6.000000,
        '10101001':    6.000000,
        '11101001':    6.000000,
        '10111001':    6.000000,
        '11111001':    6.000000,
        '01010101':    6.000000,
        '11011101':    6.000000,
        '00000011':    6.000000,
        '10000011':    6.000000,
        '01000011':    6.000000,
        '00100011':    6.000000,
        '10100011':    6.000000,
        '00010011':    6.000000,
        '01010011':    6.000000,
        '11111111':    6.000000,
        '00110000':   11.000000,
        '11110000':   11.000000,
        '10011000':   11.000000,
        '11011000':   11.000000,
        '01100100':   11.000000,
        '11100100':   11.000000,
        '10001100':   11.000000,
        '01001100':   11.000000,
        '00110010':   11.000000,
        '01110010':   11.000000,
        '10011010':   11.000000,
        '01011010':   11.000000,
        '00100110':   11.000000,
        '10100110':   11.000000,
        '11100110':   11.000000,
        '01001110':   11.000000,
        '00110001':   11.000000,
        '10110001':   11.000000,
        '00011001':   11.000000,
        '01011001':   11.000000,
        '11011001':   11.000000,
        '01100101':   11.000000,
        '10001101':   11.000000,
        '11001101':   11.000000,
        '10110011':   11.000000,
        '01110011':   11.000000,
        '11110011':   11.000000,
        '00011011':   11.000000,
        '10011011':   11.000000,
        '00100111':   11.000000,
        '01100111':   11.000000,
        '00001111':   11.000000,
        '11001111':   11.000000,
        '01101100':   17.000000,
        '10011100':   17.000000,
        '00111100':   17.000000,
        '11000110':   17.000000,
        '10010110':   17.000000,
        '11001001':   17.000000,
        '01101001':   17.000000,
        '00111001':   17.000000,
        '10011101':   17.000000,
        '11000011':   17.000000,
        '01100011':   17.000000,
        '10010011':   17.000000,
        '11001100':   23.000000,
        '01100110':   23.000000,
        '00110110':   23.000000,
        '10011001':   23.000000,
        '00110011':   23.000000 
    }

    # Minimum and maximum values are ok
    for bstring, energy in bitstrings.items():
        eigval = ham.test_bitstring(bstring)
        assert eigval == energy


def test_sumstring_2D():

    ham = TurnCircuitHamiltonian2D('HPPHP')

    for i in range(5):
        for j in range(i+1, 5):

            sumstring = {
                'x+': ham.sum_string(i, j, 'x+'),
                'x-': ham.sum_string(i, j, 'x-'),
                'y+': ham.sum_string(i, j, 'y+'),
                'y-': ham.sum_string(i, j, 'y-')
            }

        for bstring in


def test_interaction_2D():

prefactor = qand([
    qxnor(sumstring['x+'][r],
          sumstring['x-'][r])
for r in range(maximum)])

first_qxor = qxor(sumstring['y+'][0], sumstring['y-'][0])
first_qand = qand([
    qxnor(sumstring['y+'][r],
          sumstring['y-'][r])
for r in range(1, maximum)]) 
correction = qxnor(sumstring['y+'][0],
                   sumstring['y-'][0])
final_sum = sum([
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
for p in range(2, maximum+1)])

ham.expr = prefactor
print('prefactor : %d' % ham.test_bitstring(bitstring))

ham.expr = first_qxor
print('first_qxor : %d' % ham.test_bitstring(bitstring))

ham.expr = first_qand
print('first_qand : %d' % ham.test_bitstring(bitstring))

ham.expr = final_sum
print('final_sum : %d' % ham.test_bitstring(bitstring))

ham.expr = data.a_y(i, j)
hit = ham.test_bitstring(bitstring) if data.expr != 0 else 0
print('total a_y : %d' % hit)

print('FINAL SUM DECOMPOSED')
for p in range(1, maximum):

    print('  p = %d' % p)

    first_term = qxor(sumstring['y+'][p-1],
                      sumstring['y+'][p])
    second_term = qand([
                qxnor(sumstring['y+'][r],
                      sumstring['y+'][r+1])
            for r in range(p-2)])
    third_term = qand([
                qxor(sumstring['y+'][r],
                     sumstring['y-'][r])
            for r in range(p)])
    fourth_term = qand([
                qxnor(sumstring['y+'][r],
                      sumstring['y-'][r])
            for r in range(p+1, maximum-1)])

    ham.expr = first_term
    hit = ham.test_bitstring(bitstring) if type(data.expr) is not int else data.expr
    print('    first_term : %d' % hit)

    ham.expr = second_term
    hit = ham.test_bitstring(bitstring) if type(data.expr) is not int else data.expr
    print('    second_term : %d' % hit)

    ham.expr = third_term
    hit = ham.test_bitstring(bitstring) if type(data.expr) is not int else data.expr
    print('    third_term : %d' % hit)

    ham.expr = fourth_term
    hit = ham.test_bitstring(bitstring) if type(data.expr) is not int else data.expr
    print('    fourth_term : %d' % hit)

    print('    DECOMPOSITION SECOND TERM')
    for r in range(p-2):

        term = qxnor(sumstring['y+'][r],
                     sumstring['y-'][r+1])

        ham.expr = term
        hit = ham.test_bitstring(bitstring) if type(data.expr) is not int else data.expr
        print('    -> term %d : %d' % (r, hit))

print('Steric')
for i in range(6):
    for j in range(i+1, 6):

        ham.expr = data.overlap(i, j)
        hit = ham.test_bitstring(bitstring) if data.expr != 0 else 0
        print('steric (%d, %d)  =   %d' % (i, j, hit))

print('A_X and A_Y')
for i in range(6):
    for j in range(i+1, 6):

        ham.expr = data.a_x(i, j)
        hit = ham.test_bitstring(bitstring) if data.expr != 0 else 0
        print('x (%d, %d)  =   %d' % (i, j, hit))

        ham.expr = data.a_y(i, j)
        hit = ham.test_bitstring(bitstring) if data.expr != 0 else 0
        print('y (%d, %d)  =   %d' % (i, j, hit))

print('BITSTRINGS')
ham.expr = data.interaction_term()
for bstring in bitstrings:
    print('%s : %d' % (bstring, ham.test_bitstring(bstring)))
