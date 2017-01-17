import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

from kernels import LinearKernel, GaussianKernel, MultiChannelGaussianKernel
from filters import simple_klms, delayed_klms, exo_klms

if __name__ == "__main__":
    mat = loadmat("../data/bicycle_data.mat")

    x_noise = mat['x_noise'][0]  # current signal
    y_noise = mat['y_noise'][0]  # voltage signal
    z_noise = mat['z_noise'][0]  # temperature signal
    a_noise = mat['a_noise'][0]  # altitude signal

    y_a_noise = np.concatenate((y_noise.reshape(-1, 1), a_noise.reshape(-1, 1)), axis=1)

    t0 = time.time()
    f, a, e = exo_klms(y_a_noise, y_noise, MultiChannelGaussianKernel(sigmas=(6.42, 25.18)), learning_rate=1, delay=50, sparsify=(0.98, 0.5))
    print('Elapsed time: {0:.2f} secs'.format(time.time() - t0))

    # f, a, e = exo_klms(y_a_noise, y_noise, MultiChannelGaussianKernel(sigmas=(6.42, 25.18)), learning_rate=1, delay=50)
    # f, a, e = delayed_klms(y_a_noise, y_noise, MultiChannelGaussianKernel(sigmas=(6.42, 25.18)), learning_rate=0.1, delay=50)
    # f, a, e = delayed_klms(y_noise, y_noise, GaussianKernel(sigma=5), learning_rate=0.1, delay=50)
    # f, a = simple_klms(y_noise, y_noise, GaussianKernel(sigma=5), learning_rate=0.1)

    plt.figure()
    plt.plot(y_noise, 'ro')
    plt.plot(f, 'b-', linewidth=2.0)

    plt.figure()
    plt.plot(y_noise, 'ro')
    plt.plot(f, 'b-', linewidth=3.0)
    plt.xlim((0, 500))
    plt.ylim((1.0, 2.6))

    plt.show()

    print("done")
