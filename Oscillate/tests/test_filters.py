# encoding: utf-8


"""
Test the filters

"""

import sys
sys.path.append("../..")

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi, linalg
from scipy import fftpack as fftp

from Optolib import *
from Optolib.discrete import *
from Optolib.components import *

def signal_difference(ds1, u1, ds2, u2, label=''):
    signal2 = ds2.as_signal(ds1, u2)
    err =  linalg.norm(u1 - signal2)/linalg.norm(u1)
    print "Error in %s signal: %.2g" % (label,err)

# normalized parameters:
w0 = 2*pi

T = 25.0
N = 1024

Nf = 2
Nms = 64
Tms = 100.0
meps = 1e-6

pl.clf()

dsi = DiscreteSignal(N, T)

psi = PeriodicSignal(Nf, w0, real=True)
pso = PeriodicSignal(Nf, w0)

msi = MultiscaleSignal(Nms, Tms, Nf, w0, real=True)
mso = MultiscaleSignal(Nms, Tms, Nf, w0)

dw = msi.omega[0,4]
A0 = 1.0

#Input signal
u_in = A0 * np.cos((w0+dw)*dsi.t)

#Input signal for periodic signal
v0_in = np.zeros(psi.N, np.complex_)
v0_in[1] = A0

#Input signal for multiscale signal
vm_in = np.zeros(msi.N, np.complex_)
vm_in[1] = np.exp(1j*dw*msi.t[1])

#Check signals
signal_difference(dsi, u_in, psi, v0_in, 'periodic')
signal_difference(dsi, u_in, msi, vm_in, 'multiscale')

#Signal power
print "Signal", dsi.power(u_in)
print "Periodic", psi.power(v0_in)
print "Multiscale", msi.power(vm_in)

# Create filters




