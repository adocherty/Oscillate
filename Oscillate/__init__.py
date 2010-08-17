# encoding: utf-8
"""
Oscillib
========

Library for modelling the phase noise characteristics of opto-electronic oscillators
in many different configurations

-----

Copyright © 2010 Andrew Docherty

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
__version__ = '0.1'

from scipy import fftpack as fftp

# Common components
import discrete, analysis,  components

# Addon modules
import modulators, detectors, amplifiers, noisesources, opticalfiber

# Other modules that should be loaded for 'from OEO import *':
#__all__ = ['discrete', 'modulators']

#Easy configure logging
import logging
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(levelname).1s: %(message)s')

# This is needed in the current development version of matplotlib to suppress
# the many error messages
np.seterr(invalid='ignore')

#Easy units
p_ = 1e-12
n_ = 1e-9
u_ = 1e-6
k_ = 1e3
M_ = 1e6
G_ = 1e9

# db conversion finctions
def dB_to_signal(dBi):
    return 10**(dBi/20.0)
def dB_to_power(dBi):
    return 10**(dBi/10.0)
def dBm_to_power(dBi):
    return 10**(dBi/10.0) * 1e-3

def ratio_to_dB(r):
    return 20.*np.log10(r)
def power_to_dB(P):
    return 10.*np.log10(P)
def power_to_dBm(P):
    return 10.*np.log10(P/1e-3)


#Some helper functions
def save_data(data, filename="test.dat"):
    from cPickle import dump
    f = open(filename, "wb")
    dump(data, f)
    f.close()

def load_data(filename="test.dat"):
    from cPickle import load
    f = open(filename, "rb")
    data = load(f)
    f.close()
    return data

def plot_fft_shifted(x, y, *args, **kwargs):
    import pylab
    symmetric = False
    if "symmetric" in kwargs:
        symmetric = kwargs.pop("symmetric")
    if symmetric and np.mod(x.shape[0],2)==0:
        pylab.plot(fftp.fftshift(x)[1:], fftp.fftshift(y)[1:], *args, **kwargs)
    else:
        pylab.plot(fftp.fftshift(x), fftp.fftshift(y), *args, **kwargs)

def plot_half(x, y, *args, **kwargs):
    Nmax = x.shape[0]//2
    import pylab
    pylab.plot(x[:Nmax], y[:Nmax], *args, **kwargs)

