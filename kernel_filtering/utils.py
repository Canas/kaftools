import numpy as np


def distance_to_dictionary(s, x):
    """Calculates distance from vector/matrix to list of vectors/matrices

    :param s: list of vector/matrices of shape (n_samples, n_delays, n_channels)
    :param x: vector of shape (n_delays, n_channels)
    :return: norm of vector
    """

    s = np.asarray(s)
    x = np.asarray([x]*len(s))

    return np.linalg.norm(s - x, axis=1)