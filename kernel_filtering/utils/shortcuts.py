import time
import re

import matplotlib.pyplot as plt


def plot_series(data, prediction, **kwargs):
    """Shortcut to plot 2D series estimate vs target """

    plt.figure()
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])

    markersize = kwargs.pop('markersize', 5.0)
    linewidth = kwargs.pop('linewidth', 2.0)

    plt.plot(data, 'ro', markersize=markersize)
    plt.plot(prediction, 'b-', linewidth=linewidth)
    plt.show()


def timeit(f):
    """Decorator for timing execution of a function. """
    def wrap(*args, **kwargs):
        regex_str = '<(\w+ [A-Za-z]*.[a-z]*)'
        regex = re.search(regex_str, f.__str__())

        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{0} took {1:.2f} secs'.format(regex.group(1), (time2-time1)))
        return ret
    return wrap