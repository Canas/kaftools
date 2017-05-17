# -*- coding: utf-8 -*-
"""
kaftools.filters
~~~~~~~~~~~~~~~~~~~~~~~~

This module provides adaptive kernel filtering classes.
Currently supports:
- Kernel Least Mean Squeares (KLMS)
- Exogenous Kernel Least Mean Squares (KLMS-X)
- Kernel Recursive Least Squares (KRLS)

Not all filters support all features. Be sure to check the info sheet
for detailed comparisons.
"""

import numpy as np

from kaftools.kernels import GaussianKernel, MultiChannelGaussianKernel
from kaftools.utils.shortcuts import timeit


class Filter:
    """Base class for filter implementations. """

    def __init__(self, x, y, **kwargs):
        if x.ndim > 1 and x.shape[-1] == 1:
            self.x = x.ravel()
        else:
            self.x = x

        if y.ndim > 1 and y.shape[-1] == 1 or y.ndim == 1:
            self.y = y.ravel()
        else:
            raise Exception("Target data must not be multidimensional.")

        self.n = len(x)

        self.regressor = None
        self.support_vectors = None
        self.coefficients = None
        self.similarity = None
        self.kernel = None
        self.error = None
        self.learning_rate = None
        self.delay = None
        self.coefficient_history = None
        self.error_history = None
        self.param_history = None

        self._estimate = np.zeros(y.shape)

    @property
    def estimate(self):
        if self._estimate is None:
            raise AttributeError("Filter has not been applied yet.")
        else:
            return self._estimate


class KlmsFilter(Filter):
    """Original SISO KLMS filter with optional delayed input. """

    def fit(self, kernel=GaussianKernel(sigma=1.0), learning_rate=1.0, delay=1,
            kernel_learning_rate=None, sparsifiers=None, **kwargs):
        """Fit data using KLMS algorithm.

        :param kernel: Kernel class object
        :param learning_rate: float with learning rate (eta)
        :param delay: optinonal number of delayed samples to use
        :param kernel_learning_rate: float with param learning rate (mu)
        :param sparsifiers: list with Sparsifier class objects
        :return: None
        """

        if delay >= len(self.x):
            raise Exception("Delay greater than the length of the input.")

        self.learning_rate = learning_rate
        self.delay = delay

        self.coefficients = kwargs.get('coefs', np.array([self.y[delay]]))
        self.support_vectors = kwargs.get('dict', np.array([self.x[0:delay]]))

        self.coefficient_history = [self.coefficients]

        self.error_history = [0] * delay
        self.kernel = kernel
        self.param_history = [kernel.params]

        freeze_dict = kwargs.get('freeze_dict', False)
        for i in range(0, self.n - delay):
            self.regressor = self.x[i:i+delay]

            self.similarity = kernel(self.support_vectors, self.regressor)
            self.estimate[i+delay] = np.dot(self.coefficients, self.similarity)

            self.error = self.y[i + delay] - self.estimate[i + delay]
            self.error_history.append(self.error)

            self.coefficients += self.learning_rate * self.error * self.similarity

            if kernel_learning_rate:
                previous_regressor = self.x[i-1:i+delay-1]
                previous_error = self.error_history[-2]

                sigmas = []
                for k, sigma in enumerate(kernel.params):
                    new_sigma = sigma + 2 * learning_rate * kernel_learning_rate * self.error * previous_error * \
                                        (np.linalg.norm(previous_regressor - self.regressor) ** 2) * \
                                        kernel(previous_regressor, self.regressor) / sigma ** 3
                    kernel.params[k] = new_sigma
                    sigmas.append(new_sigma)
                self.param_history.append(sigmas)

            if not freeze_dict:
                if sparsifiers:
                    for sparsifier in sparsifiers:
                        sparsifier.apply(self)
                else:
                    self.support_vectors = np.append(self.support_vectors, [self.regressor], axis=0)
                    self.coefficients = np.append(self.coefficients, [learning_rate * self.error])

            self.coefficient_history.append(np.array([self.coefficients]))


class KlmsxFilter(KlmsFilter):
    """Exogenous MISO KLMS-X filter. """

    def __init__(self, x, y):
        if x.ndim < 2 and x.shape[-1] > 1:
            raise Exception("KLMS-X requires at least two input sources, otherwise use KLMS.")
        else:
            super().__init__(x, y)

    def fit(self, kernel=MultiChannelGaussianKernel(sigmas=(1.0, 1.0)), learning_rate=1.0, delay=0,
            kernel_learning_rate=None, sparsifiers=None, **kwargs):

        if len(kernel.params) != self.x.shape[-1]:
            raise Exception("There must be at least one Kernel parameter per input channel.")

        super().fit(kernel=kernel, learning_rate=learning_rate, delay=delay,
                    kernel_learning_rate=kernel_learning_rate, sparsifiers=sparsifiers, **kwargs)


class KrlsFilter(Filter):
    """Original SISO KRLS filter. """

    def __init__(self, x, y):
        super().__init__(x, y)

        self.q = None
        self.h = None
        self.z = None
        self.r = None

    @timeit
    def fit(self, kernel=GaussianKernel(sigma=1.0), regularizer=1.0, sparsifiers=None):
        """Fit data using KRLS algorithm.

        :param kernel: Kernel class object
        :param regularizer: float with regularization parameter (lambda)
        :param sparsifiers: list with Sparsifier class objects
        :return: None
        """

        self.q = np.array((regularizer + kernel(self.x[0], self.x[0])) ** (-1))
        self.coefficients = np.array([self.q * self.y[0]])
        self.coefficient_history = [self.coefficients]

        self._estimate = np.zeros(self.y.shape)
        self.support_vectors = np.array([self.x[0]]).reshape(-1, 1)
        self.error_history = [0]

        self.kernel = kernel

        for i in range(1, self.n):
            self.regressor = self.x[i]

            self.h = kernel(self.support_vectors, self.regressor).T
            self.z = np.dot(self.q, self.h)
            self.r = regularizer + kernel(self.x[i], self.x[i]) - np.dot(self.z, self.h)

            self.estimate[i] = np.dot(self.h.T, self.coefficients)
            self.error = self.y[i] - self.estimate[i]
            self.error_history.append(self.error)

            self.coefficients -= self.z * self.r ** (-1) * self.error
            self.coefficient_history.append(self.coefficients)

            if sparsifiers:
                for sparsifier in sparsifiers:
                    sparsifier.apply(self)

            else:
                self.q = self.q * self.r + np.outer(self.z, self.z.T)
                q_row = np.asarray(-self.z).reshape(-1, 1).T
                q_col = np.asarray(-self.z).reshape(-1, 1)
                q_end = np.array([1]).reshape(-1, 1)

                self.q = np.append(self.q, q_row, axis=0)
                self.q = np.append(self.q, np.concatenate((q_col, q_end), axis=0), axis=1)
                self.q = self.r ** (-1) * self.q

                self.support_vectors = np.vstack((self.support_vectors, self.regressor.reshape(1, -1)))
                self.coefficients = np.append(self.coefficients, self.r ** (-1) * self.error)