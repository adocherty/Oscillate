# encoding: utf-8
"""
Test the multiscale signals against a full signal

"""

import sys
sys.path.append("../..")

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi, linalg
from scipy import fftpack as fftp

from Oscillate import *
from Oscillate.opticalfiber import *
from Oscillate.discrete import *
from Oscillate.components import *
from Oscillate.modulators import *
from Oscillate.detectors import *

# Oscillator parameters:
f_osc = 10e9                    # Design oscillation frequency
tao = fi_length*fi_b1
n_osc = round(f_osc*tao)
w0 = 2*pi*n_osc/tao

T = 100.0*tao/n_osc                 # Time window
N = 16*1024                       # Full numeric discretization
Nf = 11                           # Number of harmonics
Tms = T
Nms = 256                     # Multi-scale discretization

#Standard signal
ds = DiscreteSignal(N, T)

# Electrical signal in FD
Nms = 1
pse = MultiscaleSignal(Nms, T, Nf, w0, real=True)

#deltams = mse.t[0]==0
#mst = np.zeros(mse.shape, np.complex_)
#mst[1] = deltams

# Input signal in TD
u_in = np.cos(w0*ds.t)

#Input signal in FD
vms_in = np.zeros(pse.shape, np.complex_)
vms_in[1] = 1.0

err_in = np.abs(pse.as_signal(ds, vms_in) - u_in).max()
print "Error in input signal:", err_in

# Modulator
eom = EOM(P0=1.0, eta=1.0, alpha=0.0, Vb=3.14, Vpi=3.14)

# Check
pss,vms_out = eom.multiscale(pse, vms_in)
dss,u_out = eom.direct(ds, u_in)

err_mod = np.abs(pss.as_signal(dss, vms_out) - u_out).max()
print "Error in modulator O(1) response:", err_mod

# Optical delay line
f1 = GenericFiber(fi_loss, fi_b2, fi_gamma, fi_length)
delay = LinearOpticalPropagation([f1])
delay = NonlinearLinearOpticalPropagation([f1])

# Check
pss,vms_del = delay.multiscale(pss, vms_out)
dss,u_del = delay.direct(dss, u_out)

err_del = np.abs(pss.as_signal(dss, vms_del) - u_del).max()
print "Error in delayline O(1) response:", err_del

# Detector
rho_det = 0.8; R_det = 50
detect = DirectDetector(rho_det, R_det)

# Check
pss,vms_det = detect.multiscale(pss, vms_del)
dss,u_det = detect.direct(dss, u_del)

err_det = np.abs(pss.as_signal(ds, vms_det) - u_det).max()
print "Error in detector O(1) response:", err_det

# Filter
wf = w0
Gwidth = 20e6           #The bandwidth of the filter
filt = LorenzianFilter2(Gwidth, wf)

# Check
pss,vms_filt = filt.multiscale(pss, vms_det)
dss,u_filt = filt.direct(dss, u_det)

err_filt = np.abs(pss.as_signal(dss, vms_filt) - u_filt).max()
print "Error in filter O(1) response:", err_filt

# Plot modulator response
pl.clf()

ax1=pl.subplot(211)
ax1.plot(ds.t, pss.as_signal(dss, vms_filt), 'r--')
ax1.plot(ds.t, u_filt, 'k:')

ax2=pl.subplot(212)
Nplot = 2048
ax2.plot(pss.ms*w0, np.abs(vms_filt), 'ro')
ax2.plot(ds.omega[:Nplot], np.abs(fftp.fft(u_filt)[:Nplot]/ds.N), 'k:')
ax2.plot(ds.omega[:Nplot], np.abs(filt.transfer_function(ds.omega)[:Nplot]), 'k:')
ax2.semilogy()
ax2.grid()

pl.draw()


