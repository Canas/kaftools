import numpy as np


class Kernel:
    """Base class for Kernel products. """

    def __init__(self):
        super().__init__()

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
        super().__init__()
        self.sigma = sigma

    def __call__(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2)**2/(2*self.sigma**2))


class MultiChannelGaussianKernel(Kernel):
    """ Multi-channel Gaussian Kernel definition."""

    def __init__(self, sigmas):
        super().__init__()
        if type(sigmas) is not np.ndarray:
            sigmas = np.array(sigmas)
        self.sigmas = sigmas

    def __call__(self, x1, x2):
        if type(x1) is list:
            x1 = np.asarray(x1)
            x2 = np.asarray([x2]*len(x1))

            distance = np.linalg.norm(x1 - x2, axis=1)
            term_sum = np.sum(distance**2/(2*self.sigmas**2), axis=1)

        else:
            distance = np.linalg.norm(x1 - x2, axis=0)
            term_sum = np.sum(distance**2/(2*self.sigmas**2))

        return np.exp(-term_sum)

    def eval(self, x1, x2):
        """Older non-vectorized implementation. """
        n_channels = len(self.sigmas)

        term_sum = 0.0
        for i in range(n_channels):
            term_sum += np.linalg.norm(x1[:, i] - x2[:, i])**2/(2*self.sigmas[i]**2)

        return np.exp(-term_sum)
