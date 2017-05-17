import matplotlib.pyplot as plt
import numpy as np

from kernel_filtering.filters import KlmsFilter
from kernel_filtering.kernels import GaussianKernel
from kernel_filtering.utils.shortcuts import plot_series, plot_squared_error
from kernel_filtering.sparsifiers import NoveltyCriterion

if __name__ == "__main__":
    # Cargar datos
    data = np.load('./data/pretrained_data_lorentz.npz')

    # sparsify lorentz : lr(1e-2), novelty(0.99919, 1.0)
    # sparsify wind: lr(1e-2), novelty(0.9934, 1.0)
    # Configurar KLMS
    klms_params = {
        'kernel': GaussianKernel(sigma=float(data['sigma_k_post'])),
        'learning_rate': 1e-1,
        'delay': int(data['delay']),
        #'sparsifiers': [NoveltyCriterion(0.99919, 1.0)]
        'coefs': data['a_post'],
        'dict': data['s_post'].T,
        'freeze_dict': True
    }
    # np.seterr(all='raise')

    klms = KlmsFilter(data['y_prog'], data['y_prog'])
    klms.fit(**klms_params)

    print(len(klms.support_vectors))
    plot_series(data['y_prog'], klms.estimate, markersize=1, linewidth=1, figsize=(15, 3))
    plot_squared_error(klms.error_history)
    import matplotlib.pyplot as plt
    #plt.semilogy(np.array(klms.error_history)**2)
    #plt.show()