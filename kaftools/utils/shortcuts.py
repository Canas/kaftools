# -*- coding: utf-8 -*-
"""
kaftools.utils.shortcuts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides utility functions that are used within
kaftools that can be useful for external scripts.
"""

import time
import re

import numpy as np
import matplotlib.pyplot as plt


def plot_series(data, prediction, **kwargs):
    """Shortcut to plot 2D series estimate vs target """

    if 'figsize' in kwargs:
        fig = plt.figure(figsize=kwargs['figsize'])
    else:
        fig = plt.figure()
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])

    markersize = kwargs.pop('markersize', 5.0)
    linewidth = kwargs.pop('linewidth', 2.0)

    plt.plot(data, 'ro', markersize=markersize)
    plt.plot(prediction, 'b-', linewidth=linewidth)
    #return fig


def plot_squared_error(error_history, **kwargs):
    sqerror = np.asarray(error_history)**2
    """Shortcut to plot squared error """

    if 'figsize' in kwargs:
        fig = plt.figure(figsize=kwargs['figsize'])
    else:
        fig = plt.figure()
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])

    linewidth = kwargs.pop('linewidth', 2.0)

    plt.semilogy(sqerror, 'b-', linewidth=linewidth)
    plt.show()
    return fig


def timeit(f):
    """Decorator for timing execution of a function. """
    def wrap(*args, **kwargs):
        regex_str = '<(\w+ [A-Za-z]*.[a-z]*)'
        regex = re.search(regex_str, f.__str__())

        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{0} took {1:.2f} secs'.format(regex.group(1), (time2-time1)))
        return ret
    return wrap


def distance_to_dictionary(s, x):
    """Calculates distance from vector/matrix to list of vectors/matrices

    :param s: list of vector/matrices of shape (n_samples, n_delays, n_channels)
    :param x: vector of shape (n_delays, n_channels)
    :return: norm of vector
    """

    s = np.asarray(s)
    x = np.asarray([x]*len(s))

    return np.linalg.norm(s - x, axis=1)
