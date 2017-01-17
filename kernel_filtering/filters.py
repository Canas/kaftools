import numpy as np
from utils import distance_to_dictionary


def exo_klms(x, y, kernel, learning_rate=0.1, sparsify=None, delay=0):
    assert delay <= len(x)

    n = x.shape[0]

    estimate = np.zeros(y.shape)
    a_coef = [learning_rate * y[delay]]
    support_vectors = [x[0:delay+1]]
    error_history = [0]*delay

    for i in range(1, n-delay):
        regressor = x[i:i+delay+1]

        centers = kernel(support_vectors, regressor)
        estimate[i+delay] = np.dot(np.array(a_coef), centers)

        error = y[i + delay] - estimate[i + delay]
        error_history.append(error)

        if sparsify:
            # distance = distance_to_dictionary(support_vectors, regressor)
            distance = centers[::-1]
            if np.max(distance) <= sparsify[0] and np.abs(error) <= sparsify[1]:
                support_vectors.append(regressor)
                a_coef.append(learning_rate * error)
        else:
            support_vectors.append(regressor)
            a_coef.append(learning_rate * error)

    return estimate, a_coef, error_history


def simple_klms(x, y, kernel, learning_rate=0.1, sparsify=None):
    """ Simple KLMS for 1-channel data

    :param x: input array
    :param y: target array
    :param kernel: kernel object with params if needed
    :param learning_rate:
    :param sparsify:
    :return: lists with estimations f and coefficients a overt time
    """

    n = x.shape[0]

    f = np.zeros(y.shape)
    a = [learning_rate * y[0]]
    d = [x[0]]

    for i in range(1, n):
        for j in range(0, i):
            f[i] += a[j] * kernel(x[i], x[j])
        error = y[i] - f[i]
        d.append(x[i])
        a.append(learning_rate * error)

    return f, a


def delayed_klms(x, y, kernel, learning_rate=0.1, sparsify=None, delay=0):
    """ Simple KLMS for 1-channel data and a fixed delay

    :param x:
    :param y:
    :param kernel:
    :param learning_rate:
    :param sparsify:
    :param delay:
    :return:
    """

    assert delay <= len(x)

    n = x.shape[0]

    f = np.zeros(y.shape)
    a = [learning_rate * y[delay]]
    d = [x[0:delay+1]]
    e = [0]*delay

    for i in range(1, n - delay):
        sequence = x[i:i+delay+1]
        for j in range(0, i):
            f[i+delay] += a[j] * kernel(d[j], sequence)
        error = y[i+delay] - f[i+delay]

        e.append(error)
        d.append(sequence)
        a.append(learning_rate * error)

    return f, a, e
