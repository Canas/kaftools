import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt


def gram(x1, x2):
    return x1[:].dimshuffle([0, 'x', 1]) - x2[:].dimshuffle(['x', 0, 1])


def logp_mean(x, sv, alpha, sigma_k):
    """
    N >= n
    x: shape(N,1)
    sv: shape(n, 1)
    alpha: shape(n, 1)
    return: shape(N, 1)
    """
    return np.dot(alpha, kern_mat(x, sv, sigma_k).T)


def kern_mat(x1, sv, sigma_k):
    #temp = gram(x1, sv)**2
    temp = np.subtract.outer(x1, sv)**2
    return np.exp(-1/(2*sigma_k**2)*temp)
    # return tt.exp(-1/(2*sigma_k**2)*gram(x, sv)**2)


def logp_d(sv, sigma_l, sigma_k):
    return -0.5*tt.log(2*np.pi*sigma_l**2) - 1/(2*sigma_l**2)*(norm_gram(sv, sigma_k)**2)


def norm_gram(sv, sigma_k):
    return tt.sum(tt.exp(-1/(2*sigma_k**2)*np.subtract.outer(sv, sv)**2))


if __name__ == "__main__":
    # Generate Lorentz
    t = np.linspace(0, 0.1, 1000)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    z = np.zeros(len(t))

    x[0] = 1
    y[0] = 1
    z[0] = 1

    sigmaL = 10
    rhoL = 28
    betaL = 8 / 3
    sL = 0.01

    for i in range(1, len(t)):
        x[i] = x[i - 1] + sL * (sigmaL * (y[i - 1] - x[i - 1]))
        y[i] = y[i - 1] + sL * (x[i - 1] * (rhoL - z[i - 1]) - y[i - 1])
        z[i] = z[i - 1] + sL * (x[i - 1] * y[i - 1] - betaL * z[i - 1])

    signal = [x, y, z]

    # PyMC3
    from pymc3 import Model, HalfNormal, Normal, Uniform, Flat, Potential, Slice, Deterministic, sample

    model = Model()

    data = x[0:50]
    target = x[1:51]
    dict_size = 10
    with model:
        sigma_e = HalfNormal('sigma_e', sd=1.)
        sigma_k = HalfNormal('sigma_k', sd=1.)
        sigma_l = HalfNormal('sigma_l', sd=1.)

        s = Uniform('s', np.min(data), np.max(data), shape=dict_size)
        # a = Normal('a', sd=1., shape=dict_size)
        a = Flat('a', shape=dict_size)

        # tdata = Deterministic('data', data)
        # tdata = theano.shared(np.asarray(data, dtype=theano.config.floatX))

        logp_dict = Potential('logp_dict', logp_d(s, sigma_l, sigma_k))
        mean = logp_mean(data, s, a, sigma_k)
        logp_like = Normal('like', mu=mean, sd=sigma_e, observed=target)

        # step = Metropolis()
        # trace = sample(2000, step)
        trace = sample(2000)