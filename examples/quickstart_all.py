import numpy as np
from scipy.io import loadmat

from kernel_filtering.filters import KlmsFilter, KlmsxFilter, KrlsFilter
from kernel_filtering.kernels import GaussianKernel, MultiChannelGaussianKernel
from kernel_filtering.sparsifiers import NoveltyCriterion, ApproximateLinearDependency
from kernel_filtering.utils.shortcuts import plot_series

if __name__ == "__main__":
    # KLMS data
    mat = loadmat("data/bicycle_data.mat")
    x_noise = mat['x_noise'][0]  # current signal
    y_noise = mat['y_noise'][0]  # voltage signal
    z_noise = mat['z_noise'][0]  # temperature signal
    a_noise = mat['a_noise'][0]  # altitude signal
    y_a_noise = np.concatenate((y_noise.reshape(-1, 1), a_noise.reshape(-1, 1)), axis=1) # [v, a]

    # KRLS data
    mat = loadmat("data/data.mat")
    voltage_discharge_krr = [voltage_cycle[0] for voltage_cycle in mat['voltage_resample_krr'][0]]
    energy_discharge_krr = [energy_cycle[0] for energy_cycle in mat['energy_resample_krr'][0]]

    klms_params = {
        'learning_rate': 0.1,
        'kernel': GaussianKernel(5.0),
        'delay': 30,
        'sparsifiers': [NoveltyCriterion(0.975, 1.0)],
        'kernel_learning_rate': 1e4
    }

    klmsx_params = {
        'learning_rate': 0.02,
        'kernel': MultiChannelGaussianKernel(sigmas=(6.42, 25.18)),
        'delay': 30,
        'sparsifiers': [NoveltyCriterion(0.975, 1.0)],
        'kernel_learning_rate': 1e4
    }

    krls_params = {
        'kernel': GaussianKernel(sigma=2.0),
        'regularizer': 0.1,
        'sparsifiers': [ApproximateLinearDependency(1e-2)]
    }

    klms = KlmsFilter(y_noise, y_noise)
    klmsx = KlmsxFilter(y_a_noise, y_noise)
    krls = KrlsFilter(voltage_discharge_krr[0], voltage_discharge_krr[0])

    klms.fit(**klms_params)
    klmsx.fit(**klmsx_params)
    krls.fit(**krls_params)

    klms_plot = {
        'title': 'KLMS on Bycicle data; {0} support vectors'.format(len(klms.support_vectors)),
    }

    klmsx_plot = {
        'title': 'KLMS-X on Bycicle data; {0} support vectors'.format(len(klmsx.support_vectors)),
    }

    krls_plot = {
        'title': 'KRLS on NASA data; {0} support vectors'.format(len(krls.support_vectors)),
        'xlim': (40, 200),
        'ylim': (2.0, 4.0),
        'linewidth': 3.0
    }

    plot_series(y_noise, klms.estimate, **klms_plot)
    plot_series(y_noise, klmsx.estimate, **klmsx_plot)
    plot_series(voltage_discharge_krr[0], krls.estimate, **krls_plot)
