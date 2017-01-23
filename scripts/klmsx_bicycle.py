import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from kernel_filtering.kernels import LinearKernel, GaussianKernel, MultiChannelGaussianKernel
from kernel_filtering.filters import simple_klms, delayed_klms, klmsx

if __name__ == "__main__":
    mat = loadmat("../data/bicycle_data.mat")

    x_noise = mat['x_noise'][0]  # current signal
    y_noise = mat['y_noise'][0]  # voltage signal
    z_noise = mat['z_noise'][0]  # temperature signal
    a_noise = mat['a_noise'][0]  # altitude signal

    y_a_noise = np.concatenate((y_noise.reshape(-1, 1), a_noise.reshape(-1, 1)), axis=1)

    t0 = time.time()
    f, a, e = klmsx(y_a_noise, y_noise, MultiChannelGaussianKernel(sigmas=(6.42, 25.18)), learning_rate=0.02, delay=30, sparsify=(0.975, 1))
    # f, a, e = klmsx(y_a_noise, y_noise, MultiChannelGaussianKernel(sigmas=(6.42, 25.18)), learning_rate=1, delay=50)

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


    # SPECIAL PLOTS

    # Transient
    fsize = 26
    fig = plt.figure(101, figsize=(16, 4))
    ax = plt.gca()
    plt.plot(f, 'b', label='KLMS-X predictions', linewidth=4)
    plt.plot(y_noise, color='red', marker='.', linestyle='None', ms=8, label='true voltage measurements')
    leg = plt.legend(ncol=4, frameon=False, shadow=True, loc=9, prop={'size': fsize})
    frame = leg.get_frame()
    frame.set_facecolor('0.9')
    plt.xlabel('time [sample]', size=fsize)
    plt.ylabel('voltage', size=fsize)
    plt.axis('tight')
    plt.rc('xtick', labelsize=fsize)
    plt.rc('ytick', labelsize=fsize)
    plt.title('One-step-ahead prediction of voltage signal (KLMS-X)', size=fsize)  # ,weight='bold')
    plt.show()

    # Steady-Transient
    fsize = 26
    fig = plt.figure(101, figsize=(16, 4))
    ax = plt.gca()
    plt.plot(f, 'b', label='KLMS-X predictions', linewidth=4)
    plt.plot(y_noise, color='red', marker='.', linestyle='None', ms=8, label='true voltage measurements')
    leg = plt.legend(ncol=4, frameon=False, shadow=True, loc=9, prop={'size': fsize})
    frame = leg.get_frame()
    frame.set_facecolor('0.9')
    plt.xlabel('time [sample]', size=fsize)
    plt.ylabel('voltage', size=fsize)
    #  plt.title('title', size = fsize)
    #  plt.axis('tight')
    plt.rc('xtick', labelsize=fsize)
    plt.rc('ytick', labelsize=fsize)
    plt.title('KLMS-X prediction: transition to steady-state region', size=fsize)  # ,weight='bold')
    plt.axis([750, 1250, 0, 1.6])
    plt.show()

    # Steady
    fsize = 26
    fig = plt.figure(101, figsize=(16, 4))
    ax = plt.gca()
    plt.plot(f, 'b', label='KLMS-X predictions', linewidth=4)
    plt.plot(y_noise, color='red', marker='.', linestyle='None', ms=8, label='true voltage measurements')
    leg = plt.legend(ncol=4, frameon=False, shadow=True, loc=9, prop={'size': fsize})
    frame = leg.get_frame()
    frame.set_facecolor('0.9')
    plt.xlabel('time [sample]', size=fsize)
    plt.ylabel('voltage', size=fsize)
    #  plt.title('title', size = fsize)
    plt.axis('tight')
    plt.rc('xtick', labelsize=fsize)
    plt.rc('ytick', labelsize=fsize)
    plt.title('KLMS-X prediction: steady-state region', size=fsize)  # ,weight='bold')
    plt.axis([2000, 2500, -0.9, 0.2])
    plt.show()

