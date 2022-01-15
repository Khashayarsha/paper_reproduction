# %%
# %reset -f
import pandas as pd
import numpy as np
import bivariate_poisson
import os


def produce_selection_matrix(teams_in_match, N=33):
    """ returns the selection-matrix that selects (a_i, a_j, B_i, B_j)' when
        post-multiplied with f_t
        f_ijt = selec_mat(i,j,N) * f_t """
    i, j = teams_in_match
    M = np.zeros((4, 2*N))
    M[:, i] = np.array((1, 0, 0, 0))
    M[:, j] = np.array((0, 1, 0, 0))
    M[:, N+i] = np.array((0, 0, 1, 0))
    M[:, N+j] = np.array((0, 0, 0, 1))

    return M


def construct_selection_mat_dict(n):
    """Returns a dictionary with 'keys' being (home, away)
     and 'values' being the corresponding selection matrix M_ij. 
     As to not construct them ad-hoc while calculating likelihood """
    combinations = []
    for i in range(n):
        for j in range(n):
            if i != j:
                combinations.append((i, j))
    selection_dictionary = dict(
        zip(combinations, map(produce_selection_matrix, combinations)))
    return selection_dictionary
