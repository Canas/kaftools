import time

import matplotlib.pyplot as plt
from scipy.io import loadmat

from kernel_filtering.filters import simple_krls
from kernel_filtering.kernels import GaussianKernel

if __name__ == "__main__":
    mat = loadmat("../data/raw_data.mat")

    voltage_discharge_krr = [voltage_cycle[0] for voltage_cycle in mat['voltage_resample_krr'][0]]
    energy_discharge_krr = [energy_cycle[0] for energy_cycle in mat['energy_resample_krr'][0]]

    x = voltage_discharge_krr[0]
    y = voltage_discharge_krr[0]

    t0 = time.time()
    estimates, coefs, errors = simple_krls(x, y, GaussianKernel(sigma=1.0), regularizer=1.0)
    print("Elapsed time: {0:.2f} seconds".format(time.time() - t0))

    plt.figure()
    plt.plot(y, 'ro')
    plt.plot(estimates, 'b-', linewidth=3.0)
    plt.xlim((40, 200))
    plt.ylim((2.0, 4.0))

    plt.show()

    print("Done")
