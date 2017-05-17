import numpy as np
from scipy.io import loadmat

from kaftools.filters import KlmsxFilter
from kaftools.kernels import MultiChannelGaussianKernel
from kaftools.sparsifiers import NoveltyCriterion
from kaftools.utils.shortcuts import plot_series

if __name__ == '__main__':
    # Cargar datos
    mat = loadmat("data/bicycle_data.mat")
    y_noise = mat['y_noise'][0]  # voltage signal
    a_noise = mat['a_noise'][0]  # altitude signal
    y_a_noise = np.concatenate((y_noise.reshape(-1, 1), a_noise.reshape(-1, 1)), axis=1)  # [v, a]

    # Kernel compartido
    kernel = MultiChannelGaussianKernel(sigmas=(6.42, 25.18))

    # Configurar KLMS-X
    klmsx_params = {
        'learning_rate': 0.02,
        'kernel': kernel,
        'delay': 30,
        'sparsifiers': [NoveltyCriterion(0.975, 1.0)],
        'kernel_learning_rate': 1e4
    }
    klmsx = KlmsxFilter(y_a_noise, y_noise)
    klmsx.fit(**klmsx_params)

    # El kernel usado ya fue adaptadom, por lo que si lo usamos de nuevo, los resultados deberían cambiar
    # (ojalá para mejor)
    klmsx_2 = KlmsxFilter(y_a_noise, y_noise)
    klmsx_2.fit(**klmsx_params)

    # Graficar ambos resultados
    plot_series(y_noise, klmsx.estimate, title='Nº support vectors {0}'.format(len(klmsx.support_vectors)))
    plot_series(y_noise, klmsx_2.estimate, title='Nº support vectors {0}'.format(len(klmsx_2.support_vectors)))
