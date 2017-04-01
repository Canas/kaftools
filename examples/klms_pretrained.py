import matplotlib.pyplot as plt
import numpy as np

from kernel_filtering.filters import KlmsFilter
from kernel_filtering.kernels import GaussianKernel
from kernel_filtering.utils.shortcuts import plot_series

if __name__ == "__main__":
    # Cargar datos
    data = np.load('./data/pretrained_data_wind.npz')

    # Configurar KLMS
    klms_params = {
        'kernel': GaussianKernel(sigma=float(data['sigma_k_post'])),
        'learning_rate': 5e-4,
        'delay': int(data['delay']),
        'coefs': data['a_post'],
        'dict': data['s_post'].T
    }
    np.seterr(all='raise')

    klms = KlmsFilter(data['y_prog'], data['y_prog'])
    klms.fit(**klms_params)

    print(len(klms.support_vectors))
    plot_series(data['y_prog'], klms.estimate, markersize=1, linewidth=1, figsize=(15, 3))