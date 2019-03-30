"""

prot_plot.py

Plots a peptide lattice model.

"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from wrappers import *
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection


def plot(string, encoding, N, fig=None, ax=None,
         colouring=None):
    """Plots a lattice protein model."""

    if   encoding == 'bead':
        peptide = process_bead(string, N)
    elif encoding == 'turn':
        peptide = process_turn(string, N)
    elif encoding == 'diamond':
        peptide = process_diamond(string, N)
    else:
        raise ValueError('Encoding unrecognised.')

    peptide += np.array([2.0, 2.0])
    draw_peptide(peptide, radius=0.2, colouring=colouring,
                 int_list=None, fig=fig, ax=ax)



def draw_peptide(peptide, save_fig=None, radius=0.2,
                 colouring=None, int_list=None, fig=None,
                 ax=None):
    """Returns a depiction of the peptide."""

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    n_residues = len(peptide)

    # Draw grid
    for i in range(n_residues):
        for j in range(n_residues):
            ax.plot(i+1, j+1, color='k', marker='o', ls='',
                    markersize=4)

    # Draw beads
    if not colouring:
        colouring = ['b' for _ in range(n_residues)]
    for i in range(n_residues):
        ax.add_patch(Circle(peptide[i, :], radius=radius,
                            edgecolor='k', facecolor=colouring[i]))


    # Draw primary structure
    for i in range(n_residues-1):

        # The lines should not intersect the balls:
        # correct for this
        dist_x = np.abs(peptide[i, 0] - peptide[i+1, 0])
        dist_y = np.abs(peptide[i, 1] - peptide[i+1, 1])

        if dist_x == 0.0:
            x_r = 0.0
        else:
            x_r = radius

        if dist_y == 0.0:
            y_r = 0.0
        else:
            y_r = radius

        # Assign the start and end points for the line
        x_0 = np.min([peptide[i, 0], peptide[i+1, 0]]) + x_r
        x_f = np.max([peptide[i, 0], peptide[i+1, 0]]) - x_r
        y_0 = np.min([peptide[i, 1], peptide[i+1, 1]]) + y_r
        y_f = np.max([peptide[i, 1], peptide[i+1, 1]]) - y_r

        ax.plot([x_0, x_f], [y_0, y_f], color='k')


    # Draw non-covalent interactions
    if int_list is not None:
        for i, j in int_list:

            # Check that distance is 1
            dist_x = np.abs(peptide[i, 0] - peptide[j, 0])
            dist_y = np.abs(peptide[i, 1] - peptide[j, 1])

            if dist_x + dist_y != 1:
                warnings.warn('L1-distance between {:d} and {:d} is not 1.0'
                              .format(i+1, j+1))
                continue

            # Draw interaction
            if dist_x == 0.0:
                y_0 = np.min([peptide[i, 1], peptide[j, 1]])
                assert peptide[i, 0] == peptide[j, 0]
                x_c = peptide[i, 0]

                y_1 = y_0 + 2/7
                y_2 = y_0 + 3/7
                y_3 = y_0 + 4/7

                ax.plot([x_c-0.2, x_c+0.2], [y_1, y_1], color='c')

            else:
                x_0 = np.min([peptide[i, 0], peptide[j, 0]])
                assert peptide[i, 1] == peptide[j, 1]
                y_c = peptide[i, 1]

                x_1 = x_0 + 0.35
                x_2 = x_0 + 0.50
                x_3 = x_0 + 0.65

                ax.plot([x_1, x_1], [y_c-0.2, y_c+0.2], color='c')
                ax.plot([x_2, x_2], [y_c-0.2, y_c+0.2], color='c')
                ax.plot([x_3, x_3], [y_c-0.2, y_c+0.2], color='c')


    ax.axis('equal')
    ax.axis('off')


#if __name__ == '__main__':
#
#    exit()
#    with open('tuan-hpph.pubo.out') as f:
#        data = [x.split()[0] for x in f.readlines()]
#
#    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(8, 8))
#    axes = [x for axlist in axes for x in axlist]
#
#    for i in range(25):
#
#        plot(data[i], 'turn', 4, fig=fig, ax=axes[i],
#             colouring=['g', 'r', 'r', 'g'])
#
#    plt.show()
#
