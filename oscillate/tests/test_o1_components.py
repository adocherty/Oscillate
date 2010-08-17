# encoding: utf-8
"""
Test components for purely periodic behaviour using order1, multiscale and direct methods

"""

import sys
sys.path.append("../..")

import unittest
from numpy.testing import *

import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi, linalg
from scipy import fftpack as fftp

from oscillate import *
from oscillate.opticalfiber import *
from oscillate.discrete import *
from oscillate.components import *
from oscillate.modulators import *
from oscillate.amplifiers import *
from oscillate.detectors import *
from oscillate.noisesources import *
from oscillate.analysis import *


Nf = 2
Nms = 256
Tms = 100.0
w0 = 2*pi*100

# Modulator & Detector
eta = 0.8
Vpi = Vb = 3.14
Vph = 50
loss = 0.9
modet = ModulatorDetector(eta, Vb, Vpi, Vph, iloss=loss)

# Filter
wf = w0
Gwidth = 10.0
filt = LorenzianFilter(wf, Gwidth)

# Amplifier
Ga = 7.5
amp = Amplifier(Ga)

# Saturable amplifier
Ga = 7.5
Psat = 0.1
samp = SimpleSaturableAmplifier(Ga, Psat)


def signal_spectrum(pss, loop):
    # The initial signal
    v_sig = np.ones(pss.shape, np.complex_)
    v_in = v_sig.copy()

    ii=0
    difference=np.inf
    while (ii<1) and (difference>1e-12):
        pss,v_sig = loop.order1(pss,v_in)
        
        difference = linalg.norm(v_in-v_sig)
        v_in = v_sig
        
        ii+=1

    Ps = pss.power(v_sig)
    print "Converged to %.4g in %d iterations, signal power P=%.3g" % (difference,ii,Ps)
    print "Loop coefficients", np.abs(v_sig[:4])


class TestO1(TestCase):
    def _multiscale(self, mss, comp):
        v_in = np.ones(mss.shape, np.complex_)
        mso,v_out = comp.multiscale(mss,v_in)
        print "MS:", v_out[:,0]
        return mso, v_out

    def _order1(self, pss, comp):
        v_in = np.ones(pss.shape, np.complex_)
        pso,v_out = comp.order1(pss,v_in)
        print "O1:", v_out
        return pso, v_out

    def _direct(self, dss, comp):
        v_in = np.ones(dss.shape, np.complex_)
        dso,v_out = comp.direct(mss,v_in)
        return dso, v_out

    def _check_component(self, comp, Nf, Tms, w0):
        m0s = MultiscaleSignal(1, Tms, Nf, w0, real=True)
        pss = PeriodicSignal(Nf, w0, real=True)
        
        mso,vm_out = self._multiscale(m0s, comp)
        mso,vp_out = self._order1(pss, comp)
        
        assert_array_almost_equal(vm_out[:,0], vp_out)

    def test_amplifier(self):
        self._check_component(amp, 2, Tms, w0)
        self._check_component(amp, 4, Tms, w0)
        self._check_component(amp, 6, Tms, w0)

    def test_modet(self):
        self._check_component(modet, 2, Tms, w0)
        self._check_component(modet, 4, Tms, w0)
        self._check_component(modet, 6, Tms, w0)

    def test_filter(self):
        self._check_component(filt, 2, Tms, w0)
        self._check_component(filt, 4, Tms, w0)
        self._check_component(filt, 6, Tms, w0)
        
    def test_amplifier(self):
        self._check_component(samp, 2, Tms, w0)
        self._check_component(samp, 4, Tms, w0)
        self._check_component(samp, 6, Tms, w0)


if __name__=="__main__":
    unittest.main()
