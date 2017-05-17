# -*- coding: utf-8 -*-
"""
kernel_filtering.utils.data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides utility functions that can be used to
generate synthetic datasets for use within scripts and tests.
"""

import numpy as np

def generate_lorentz_system(size=1000, step=0.1, init=(1, 1, 1), **kwargs):
    t = np.linspace(0, step, size)
    
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    z = np.zeros(len(t))

    x[0] = init[0]
    y[0] = init[1]
    z[0] = init[2]

    try:
        sigmaL = kwargs['sigmaL']
        rhoL = kwargs['rhoL']
    except KeyError as e:
        

