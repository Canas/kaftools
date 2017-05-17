import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from kaftools.filters import KlmsxFilter
from kaftools.kernels import MultiChannelGaussianKernel
from kaftools.sparsifiers import NoveltyCriterion

if __name__ == "__main__":
    # Cargar datos
    mat = loadmat("data/bicycle_data.mat")
    y_noise = mat['y_noise'][0]  # voltage signal
    a_noise = mat['a_noise'][0]  # altitude signal
    y_a_noise = np.concatenate((y_noise.reshape(-1, 1), a_noise.reshape(-1, 1)), axis=1)  # [v, a]

    # Configurar KLMS-X
    klmsx_params = {
        'learning_rate': 0.02,
        'delay': 30,
        'kernel': MultiChannelGaussianKernel(sigmas=(6.42, 25.18)),
        'sparsifiers': [NoveltyCriterion(0.975, 0.8)]
    }
    klmsx = KlmsxFilter(y_a_noise, y_noise)
    klmsx.fit(**klmsx_params)

    # Graficar resultados
    # Transient
    fsize = 26
    fig = plt.figure(101, figsize=(16, 4))
    ax = plt.gca()
    plt.plot(klmsx.estimate, 'b', label='KLMS-X predictions', linewidth=4)
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
    plt.plot(klmsx.estimate, 'b', label='KLMS-X predictions', linewidth=4)
    plt.plot(y_noise, color='red', marker='.', linestyle='None', ms=8, label='true voltage measurements')
    leg = plt.legend(ncol=4, frameon=False, shadow=True, loc=9, prop={'size': fsize})
    frame = leg.get_frame()
    frame.set_facecolor('0.9')
    plt.xlabel('time [sample]', size=fsize)
    plt.ylabel('voltage', size=fsize)
    plt.rc('xtick', labelsize=fsize)
    plt.rc('ytick', labelsize=fsize)
    plt.title('KLMS-X prediction: transition to steady-state region', size=fsize)  # ,weight='bold')
    plt.axis([750, 1250, 0, 1.6])
    plt.show()

    # Steady
    fsize = 26
    fig = plt.figure(101, figsize=(16, 4))
    ax = plt.gca()
    plt.plot(klmsx.estimate, 'b', label='KLMS-X predictions', linewidth=4)
    plt.plot(y_noise, color='red', marker='.', linestyle='None', ms=8, label='true voltage measurements')
    leg = plt.legend(ncol=4, frameon=False, shadow=True, loc=9, prop={'size': fsize})
    frame = leg.get_frame()
    frame.set_facecolor('0.9')
    plt.xlabel('time [sample]', size=fsize)
    plt.ylabel('voltage', size=fsize)
    plt.axis('tight')
    plt.rc('xtick', labelsize=fsize)
    plt.rc('ytick', labelsize=fsize)
    plt.title('KLMS-X prediction: steady-state region', size=fsize)  # ,weight='bold')
    plt.axis([2000, 2500, -0.9, 0.2])
    plt.show()
