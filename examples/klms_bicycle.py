from scipy.io import loadmat
import matplotlib.pyplot as plt

from kernel_filtering.filters import KlmsFilter
from kernel_filtering.kernels import GaussianKernel

if __name__ == "__main__":
    # Cargar datos
    mat = loadmat("data/bicycle_data.mat")
    y_noise = mat['y_noise'][0]  # voltage signal

    # Configurar KLMS
    klms_params = {
        'kernel': GaussianKernel(sigma=10),
        'learning_rate': 5e-4,
        'delay': 5,
        # 'coefs': some_coefs,
        # 'dict': some_dict
    }
    klms = KlmsFilter(y_noise, y_noise)
    klms.fit(**klms_params)

    # Graficar resultados
    # Transient
    fsize = 26
    fig = plt.figure(101, figsize=(16, 4))
    ax = plt.gca()
    plt.plot(klms.estimate, 'b', label='KLMS-X predictions', linewidth=4)
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
    plt.plot(klms.estimate, 'b', label='KLMS-X predictions', linewidth=4)
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
    plt.plot(klms.estimate, 'b', label='KLMS-X predictions', linewidth=4)
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
