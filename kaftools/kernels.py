# -*- coding: utf-8 -*-
"""
kaftools.kernels
~~~~~~~~~~~~~~~~~~~~~~~~

This module provides kernel classes for similarity evaluation.
Currently supports:
- LinearKernel
- GaussianKernel
- MultiChannelGaussianKernel

Most implementations support comparison between a list of features
against a single feature to measure distance from one point to the
rest of a dictionary.
"""

import numpy as np


class Kernel:
    """Base class for Kernel definitions. """

    def __init__(self, params=None):
        super().__init__()
        if type(params) is not np.ndarray:
            params = np.array(params).ravel()
        self.params = params

    def __call__(self, x1, x2):
        raise NotImplementedError("You must instantiate a valid Kernel object")


class LinearKernel(Kernel):
    """Linear Kernel definition. """

    def __init__(self):
        super().__init__()

    def __call__(self, x1, x2):
        return np.dot(x1, x2)


class GaussianKernel(Kernel):
    """Gaussian Kernel definition. """

    def __init__(self, sigma):
        super().__init__([sigma])
        if not sigma > 0:
            raise ZeroDivisionError("Gaussian Kernel cannot have zero or negative bandwidth.")

    def __call__(self, x1, x2):
        sigma = self.params[0]
        if x1.ndim > x2.ndim:
            x2 = np.tile(x2.reshape(1, -1), (x1.shape[0], 1))
            distance = np.linalg.norm(x1 - x2, axis=1)

        elif x2.ndim > x1.ndim:
            x1 = np.tile(x1.reshape(1, -1), (x2.shape[0], 1))
            distance = np.linalg.norm(x1 - x2, axis=1)

        else:
            x1 = x1.reshape(1, -1)
            x2 = x2.reshape(1, -1)

            distance = np.linalg.norm(x1 - x2, axis=1)

        term = distance**2/(2*sigma**2)
        return np.exp(-term)


class MultiChannelGaussianKernel(Kernel):
    """ Multi-channel Gaussian Kernel definition. For single channel use Gaussian Kernel."""

    def __init__(self, sigmas):
        super().__init__(sigmas)
        for sigma in sigmas:
            if not sigma > 0:
                raise ZeroDivisionError("Gaussian Kernel cannot have zero or negative bandwidth.")

    def __call__(self, x1, x2):
        sigmas = self.params
        if x1.ndim > x2.ndim:
            x2 = np.asarray([x2] * len(x1))

            distance = np.linalg.norm(x1 - x2, axis=1)
            term_sum = np.sum(distance**2/(2*sigmas**2), axis=1)

        elif x2.ndim > x1.ndim:
            x1 = np.asarray([x1] * len(x2))

            distance = np.linalg.norm(x1 - x2, axis=1)
            term_sum = np.sum(distance**2/(2*sigmas**2), axis=1)

        else:
            distance = np.linalg.norm(x1 - x2, axis=0)
            term_sum = np.sum(distance**2/(2*sigmas**2))

        return np.exp(-term_sum)
