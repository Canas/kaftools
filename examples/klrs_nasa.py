import time

from scipy.io import loadmat

from kernel_filtering.filters import KrlsFilter
from kernel_filtering.kernels import GaussianKernel
from kernel_filtering.sparsifiers import ApproximateLinearDependency
from kernel_filtering.utils.shortcuts import plot_series

if __name__ == "__main__":
    # Cargar datos
    mat = loadmat("data/data.mat")
    voltage_discharge_krr = [voltage_cycle[0] for voltage_cycle in mat['voltage_resample_krr'][0]]

    # Configurar KRLS
    krls_params = {
        'regularizer': 1e-1,
        'kernel': GaussianKernel(sigma=2.0),
        'sparsifiers': [ApproximateLinearDependency(threshold=1e-2)]
    }
    krls = KrlsFilter(voltage_discharge_krr[0], voltage_discharge_krr[0])
    krls.fit(**krls_params)

    # Graficar resultados
    krls_plot = {
        'title': 'KLRS on NASA data; Nº support vectors: {0}'.format(len(krls.support_vectors)),
        'xlim': (40, 200),
        'ylim': (2.0, 4.0),
        'linewidth': 3.0
    }
    plot_series(voltage_discharge_krr[0], krls.estimate, **krls_plot)