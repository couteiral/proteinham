import numpy as np


hp_set = {'H', 'P'}
mj_set = {'Y', 'C', 'K', 'E', 'L', 'V', 'R', 'P', 
          'H', 'M', 'A', 'B', 'W', 'S', 'Z', 'I', 
          'T', 'Q', 'F', 'D', 'G', 'N'}

mj_dict = {
    frozenset(['C', 'C']) : -5.440000,
    frozenset(['M', 'C']) : -4.990000,
    frozenset(['M', 'M']) : -5.460000,
    frozenset(['F', 'C']) : -5.800000,
    frozenset(['F', 'M']) : -6.560000,
    frozenset(['F', 'F']) : -7.260000,
    frozenset(['I', 'C']) : -5.500000,
    frozenset(['I', 'M']) : -6.020000,
    frozenset(['I', 'F']) : -6.840000,
    frozenset(['I', 'I']) : -6.540000,
    frozenset(['L', 'C']) : -5.830000,
    frozenset(['L', 'M']) : -6.410000,
    frozenset(['L', 'F']) : -7.280000,
    frozenset(['L', 'I']) : -7.040000,
    frozenset(['L', 'L']) : -7.370000,
    frozenset(['V', 'C']) : -4.960000,
    frozenset(['V', 'M']) : -5.320000,
    frozenset(['V', 'F']) : -6.290000,
    frozenset(['V', 'I']) : -6.050000,
    frozenset(['V', 'L']) : -6.480000,
    frozenset(['V', 'V']) : -5.520000,
    frozenset(['W', 'C']) : -4.950000,
    frozenset(['W', 'M']) : -5.550000,
    frozenset(['W', 'F']) : -6.160000,
    frozenset(['W', 'I']) : -5.780000,
    frozenset(['W', 'L']) : -6.140000,
    frozenset(['W', 'V']) : -5.180000,
    frozenset(['W', 'W']) : -5.060000,
    frozenset(['Y', 'C']) : -4.160000,
    frozenset(['Y', 'M']) : -4.910000,
    frozenset(['Y', 'F']) : -5.660000,
    frozenset(['Y', 'I']) : -5.250000,
    frozenset(['Y', 'L']) : -5.670000,
    frozenset(['Y', 'V']) : -4.620000,
    frozenset(['Y', 'W']) : -4.660000,
    frozenset(['Y', 'Y']) : -4.170000,
    frozenset(['A', 'C']) : -3.570000,
    frozenset(['A', 'M']) : -3.940000,
    frozenset(['A', 'F']) : -4.810000,
    frozenset(['A', 'I']) : -4.580000,
    frozenset(['A', 'L']) : -4.910000,
    frozenset(['A', 'V']) : -4.040000,
    frozenset(['A', 'W']) : -3.820000,
    frozenset(['A', 'Y']) : -3.360000,
    frozenset(['A', 'A']) : -2.720000,
    frozenset(['G', 'C']) : -3.160000,
    frozenset(['G', 'M']) : -3.390000,
    frozenset(['G', 'F']) : -4.130000,
    frozenset(['G', 'I']) : -3.780000,
    frozenset(['G', 'L']) : -4.160000,
    frozenset(['G', 'V']) : -3.380000,
    frozenset(['G', 'W']) : -3.420000,
    frozenset(['G', 'Y']) : -3.010000,
    frozenset(['G', 'A']) : -2.310000,
    frozenset(['G', 'G']) : -2.240000,
    frozenset(['T', 'C']) : -3.110000,
    frozenset(['T', 'M']) : -3.510000,
    frozenset(['T', 'F']) : -4.280000,
    frozenset(['T', 'I']) : -4.030000,
    frozenset(['T', 'L']) : -4.340000,
    frozenset(['T', 'V']) : -3.460000,
    frozenset(['T', 'W']) : -3.220000,
    frozenset(['T', 'Y']) : -3.010000,
    frozenset(['T', 'A']) : -2.320000,
    frozenset(['T', 'G']) : -2.080000,
    frozenset(['T', 'T']) : -2.120000,
    frozenset(['S', 'C']) : -2.860000,
    frozenset(['S', 'M']) : -3.030000,
    frozenset(['S', 'F']) : -4.020000,
    frozenset(['S', 'I']) : -3.520000,
    frozenset(['S', 'L']) : -3.920000,
    frozenset(['S', 'V']) : -3.050000,
    frozenset(['S', 'W']) : -2.990000,
    frozenset(['S', 'Y']) : -2.780000,
    frozenset(['S', 'A']) : -2.010000,
    frozenset(['S', 'G']) : -1.820000,
    frozenset(['S', 'T']) : -1.960000,
    frozenset(['S', 'S']) : -1.670000,
    frozenset(['N', 'C']) : -2.590000,
    frozenset(['N', 'M']) : -2.950000,
    frozenset(['N', 'F']) : -3.750000,
    frozenset(['N', 'I']) : -3.240000,
    frozenset(['N', 'L']) : -3.740000,
    frozenset(['N', 'V']) : -2.830000,
    frozenset(['N', 'W']) : -3.070000,
    frozenset(['N', 'Y']) : -2.760000,
    frozenset(['N', 'A']) : -1.840000,
    frozenset(['N', 'G']) : -1.740000,
    frozenset(['N', 'T']) : -1.880000,
    frozenset(['N', 'S']) : -1.580000,
    frozenset(['N', 'N']) : -1.680000,
    frozenset(['Q', 'C']) : -2.850000,
    frozenset(['Q', 'M']) : -3.300000,
    frozenset(['Q', 'F']) : -4.100000,
    frozenset(['Q', 'I']) : -3.670000,
    frozenset(['Q', 'L']) : -4.040000,
    frozenset(['Q', 'V']) : -3.070000,
    frozenset(['Q', 'W']) : -3.110000,
    frozenset(['Q', 'Y']) : -2.970000,
    frozenset(['Q', 'A']) : -1.890000,
    frozenset(['Q', 'G']) : -1.660000,
    frozenset(['Q', 'T']) : -1.900000,
    frozenset(['Q', 'S']) : -1.490000,
    frozenset(['Q', 'N']) : -1.710000,
    frozenset(['Q', 'Q']) : -1.540000,
    frozenset(['D', 'C']) : -2.410000,
    frozenset(['D', 'M']) : -2.570000,
    frozenset(['D', 'F']) : -3.480000,
    frozenset(['D', 'I']) : -3.170000,
    frozenset(['D', 'L']) : -3.400000,
    frozenset(['D', 'V']) : -2.480000,
    frozenset(['D', 'W']) : -2.840000,
    frozenset(['D', 'Y']) : -2.760000,
    frozenset(['D', 'A']) : -1.700000,
    frozenset(['D', 'G']) : -1.590000,
    frozenset(['D', 'T']) : -1.800000,
    frozenset(['D', 'S']) : -1.630000,
    frozenset(['D', 'N']) : -1.680000,
    frozenset(['D', 'Q']) : -1.460000,
    frozenset(['D', 'D']) : -1.210000,
    frozenset(['E', 'C']) : -2.270000,
    frozenset(['E', 'M']) : -2.890000,
    frozenset(['E', 'F']) : -3.560000,
    frozenset(['E', 'I']) : -3.270000,
    frozenset(['E', 'L']) : -3.590000,
    frozenset(['E', 'V']) : -2.670000,
    frozenset(['E', 'W']) : -2.990000,
    frozenset(['E', 'Y']) : -2.790000,
    frozenset(['E', 'A']) : -1.510000,
    frozenset(['E', 'G']) : -1.220000,
    frozenset(['E', 'T']) : -1.740000,
    frozenset(['E', 'S']) : -1.480000,
    frozenset(['E', 'N']) : -1.510000,
    frozenset(['E', 'Q']) : -1.420000,
    frozenset(['E', 'D']) : -1.020000,
    frozenset(['E', 'E']) : -0.910000,
    frozenset(['H', 'C']) : -3.600000,
    frozenset(['H', 'M']) : -3.980000,
    frozenset(['H', 'F']) : -4.770000,
    frozenset(['H', 'I']) : -4.140000,
    frozenset(['H', 'L']) : -4.540000,
    frozenset(['H', 'V']) : -3.580000,
    frozenset(['H', 'W']) : -3.980000,
    frozenset(['H', 'Y']) : -3.520000,
    frozenset(['H', 'A']) : -2.410000,
    frozenset(['H', 'G']) : -2.150000,
    frozenset(['H', 'T']) : -2.420000,
    frozenset(['H', 'S']) : -2.110000,
    frozenset(['H', 'N']) : -2.080000,
    frozenset(['H', 'Q']) : -1.980000,
    frozenset(['H', 'D']) : -2.320000,
    frozenset(['H', 'E']) : -2.150000,
    frozenset(['H', 'H']) : -3.050000,
    frozenset(['R', 'C']) : -2.570000,
    frozenset(['R', 'M']) : -3.120000,
    frozenset(['R', 'F']) : -3.980000,
    frozenset(['R', 'I']) : -3.630000,
    frozenset(['R', 'L']) : -4.030000,
    frozenset(['R', 'V']) : -3.070000,
    frozenset(['R', 'W']) : -3.410000,
    frozenset(['R', 'Y']) : -3.160000,
    frozenset(['R', 'A']) : -1.830000,
    frozenset(['R', 'G']) : -1.720000,
    frozenset(['R', 'T']) : -1.900000,
    frozenset(['R', 'S']) : -1.620000,
    frozenset(['R', 'N']) : -1.640000,
    frozenset(['R', 'Q']) : -1.800000,
    frozenset(['R', 'D']) : -2.290000,
    frozenset(['R', 'E']) : -2.270000,
    frozenset(['R', 'H']) : -2.160000,
    frozenset(['R', 'R']) : -1.550000,
    frozenset(['K', 'C']) : -1.950000,
    frozenset(['K', 'M']) : -2.480000,
    frozenset(['K', 'F']) : -3.360000,
    frozenset(['K', 'I']) : -3.010000,
    frozenset(['K', 'L']) : -3.370000,
    frozenset(['K', 'V']) : -2.490000,
    frozenset(['K', 'W']) : -2.690000,
    frozenset(['K', 'Y']) : -2.600000,
    frozenset(['K', 'A']) : -1.310000,
    frozenset(['K', 'G']) : -1.150000,
    frozenset(['K', 'T']) : -1.310000,
    frozenset(['K', 'S']) : -1.050000,
    frozenset(['K', 'N']) : -1.210000,
    frozenset(['K', 'Q']) : -1.290000,
    frozenset(['K', 'D']) : -1.680000,
    frozenset(['K', 'E']) : -1.800000,
    frozenset(['K', 'H']) : -1.350000,
    frozenset(['K', 'R']) : -0.590000,
    frozenset(['K', 'K']) : -0.120000,
    frozenset(['P', 'C']) : -3.070000,
    frozenset(['P', 'M']) : -3.450000,
    frozenset(['P', 'F']) : -4.250000,
    frozenset(['P', 'I']) : -3.760000,
    frozenset(['P', 'L']) : -4.200000,
    frozenset(['P', 'V']) : -3.320000,
    frozenset(['P', 'W']) : -3.730000,
    frozenset(['P', 'Y']) : -3.190000,
    frozenset(['P', 'A']) : -2.030000,
    frozenset(['P', 'G']) : -1.870000,
    frozenset(['P', 'T']) : -1.900000,
    frozenset(['P', 'S']) : -1.570000,
    frozenset(['P', 'N']) : -1.530000,
    frozenset(['P', 'Q']) : -1.730000,
    frozenset(['P', 'D']) : -1.330000,
    frozenset(['P', 'E']) : -1.260000,
    frozenset(['P', 'H']) : -2.250000,
    frozenset(['P', 'R']) : -1.700000,
    frozenset(['P', 'K']) : -0.970000,
    frozenset(['P', 'P']) : -1.750000
}

def int_matrix(pepstring):
    """Computes an interaction matrix based
    on the primary sequence of a peptide."""

    n_res    = len(pepstring)
    residues = list(pepstring)
    int_mat  = np.zeros([n_res, n_res])

    if set(residues).issubset(hp_set):

        for i in range(n_res):
            for j in range(n_res):

                if i == j:
                    continue

                if residues[i] == 'H' and residues[j] == 'H':
                    int_mat[i, j] = 1.0

    elif set(residues).issubset(mj_set):

        for i in range(n_res):
            for j in range(n_res):

                if i == j:
                    continue

                int_mat[i, j] = mj_dict[frozenset([residues[i], residues[j]])]

    else:

        raise ValueError('Encoding not recognised in %s' % pepstring)

    return int_mat
