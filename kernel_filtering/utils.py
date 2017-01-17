import numpy as np


def distance_to_dictionary(s, x):
    """Calculates distance from vector/matrix to list of vectors/matrices

    :param s: list of vector/matrices of shape n_elements x n_delay x n_inputs
    :param x: vector of shape n_delay x n_inputs
    :return:
    """

    s = np.asarray(s)
    x = np.asarray([x]*len(s))

    return np.linalg.norm(s - x, axis=1)