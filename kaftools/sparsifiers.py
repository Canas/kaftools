# -*- coding: utf-8 -*-
"""
kaftools.sparsifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides sparsifier classes for Sparsification criteria.
Currently supports:
- Novelty criterion
- Approximate Linear Dependency (ALD)

Not all filters support all sparsifiers. Be sure to check the info sheet
for detailed comparisons.
"""

import numpy as np

from kaftools.filters import KrlsFilter


class SparsifyMethod(object):
    """Base class for sparsification criteria. """

    def __init__(self, params=None):
        if type(params) is not np.ndarray:
            params = np.array(params).ravel()
        self.params = params

    def apply(self, kfilter):
        """ Abstract method for applying a concrete sparsifier over a kernel filter.

        :param kfilter: filter.Filter object
        :return: None. The object itself is modified during iterations.
        """
        pass


class NoveltyCriterion(SparsifyMethod):
    """Novelty criterion for KLMS filters. Hasn't been thoroughly tested on KRLS. """

    def __init__(self, distance_delta, error_delta):
        super().__init__([distance_delta, error_delta])

    def apply(self, kfilter):
        if np.max(kfilter.similarity) <= self.params[0] and np.abs(kfilter.error) >= self.params[1]:
            kfilter.support_vectors = np.append(kfilter.support_vectors, [kfilter.regressor], axis=0)
            kfilter.coefficients = np.append(kfilter.coefficients, [0.0])


class ApproximateLinearDependency(SparsifyMethod):
    """ALS criterion for KRLS filters. """

    def __init__(self, threshold):
        super().__init__([threshold])

    def apply(self, kfilter):
        if type(kfilter) is not KrlsFilter:
            raise Exception("ALD is only implemented for KRLS filters.")
        else:
            kernel_regressor = kfilter.kernel(kfilter.regressor, kfilter.regressor)
            kernel_support_vectors = kfilter.kernel(kfilter.support_vectors, kfilter.support_vectors)

            distance = kernel_regressor - kfilter.h ** 2 / kernel_support_vectors

            if np.min(distance) > self.params[0]:
                q_row = np.asarray(-kfilter.z).reshape(-1, 1).T
                q_col = np.asarray(-kfilter.z).reshape(-1, 1)
                q_end = np.array([1]).reshape(-1, 1)

                kfilter.q = kfilter.q * kfilter.r + np.outer(kfilter.z, kfilter.z.T)
                kfilter.q = np.append(kfilter.q, q_row, axis=0)
                kfilter.q = np.append(kfilter.q, np.concatenate((q_col, q_end), axis=0), axis=1)
                kfilter.q *= kfilter.r ** (-1)

                kfilter.coefficients = np.append(kfilter.coefficients, kfilter.r**(-1) * kfilter.error)
                kfilter.support_vectors = np.vstack((kfilter.support_vectors, kfilter.regressor.reshape(1, -1)))

            else:
                kfilter.q = kfilter.q * kfilter.r + np.outer(kfilter.z, kfilter.z.T)
                kfilter.q *= kfilter.r ** (-1)
