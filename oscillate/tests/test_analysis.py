# encoding: utf-8

"""
Test Analysis module:

Tests run:
 * 
 *

Memory checks:
 * Check memory efficiency?
 * Check memmap analysis

"""

import sys
sys.path.append("../..")

from Optolib import timer
timer.reset()

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi, linalg

from Optolib import *

from Optolib.discrete import *
from Optolib.analysis import *

# Signal
Tms = 100.0
Nms = 512
Nf = 10
w0 = 1.0

mss = MultiscaleSignal(Nms, Tms, Nf, w0, real=True)
x = np.zeros(mss.shape, np.complex_)

# Analysis
Nrt = 10000
loopsig = LoopStorage(mss, Nrt, mss.Nt)

for ii in range(Nrt):
    loopsig.store(mss, x[1])

#Analaysis
asig = Analysis()
dss,vs = loopsig.time_concatenated_signal()

#Add signal to ensemble
asig.set_signal(dss,vs)

#pn = asig.phase_noise()
f,pn = asig.phase_noise_spectrum()
f,an = asig.amplitude_noise_spectrum()
f,sigs = asig.spectrum()


