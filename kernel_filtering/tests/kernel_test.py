import unittest

import numpy as np
import numpy.testing as npt

from ..kernels import MultiChannelGaussianKernel, GaussianKernel, LinearKernel


class GaussianTests(unittest.TestCase):
    def setUp(self):
        self.x1 = np.array([1.0])
        self.x2 = np.array([1.0])
        self.kernel = GaussianKernel(sigma=1.0)

    def test_zero_distance(self):
        simmilarity = self.kernel(self.x1, self.x2)
        ground_truth = np.array([1.0])
        npt.assert_allclose(simmilarity, ground_truth)


class MultiChannelGaussianTests(unittest.TestCase):
    def setUp(self):
        self.x1 = np.array([[1.0, 1.0]])
        self.x2 = np.array([[1.0, 1.0]])
        self.kernel = MultiChannelGaussianKernel(sigmas=(1.0, 1.0))

    def test_zero_distance(self):
        simmilarity = self.kernel(self.x1, self.x2)
        ground_truth = np.array([1.0])
        npt.assert_allclose(simmilarity, ground_truth)
