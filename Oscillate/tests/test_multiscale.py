# encoding: utf-8
"""
Test the multiscale signals against a full signal

"""

import sys
sys.path.append("../..")

import unittest
from numpy.testing import *

import numpy as np
import scipy as sp

from numpy import pi, linalg
from scipy import fftpack as fftp

from Optolib import *

from Optolib.opticalfiber import *
from Optolib.discrete import *
from Optolib.components import *
from Optolib.modulators import *
from Optolib.amplifiers import *
from Optolib.detectors import *
from Optolib.noisesources import *

def signal_difference(ds1, u1, ds2, u2, label=''):
    signal2 = ds2.as_signal(ds1, u2)
    err =  linalg.norm(u1 - signal2)/linalg.norm(u1)
    print "Error in %s signal: %.2g" % (label,err)

    assert_almost_equal(err, 0)

w0 = 2*pi

T = 25.0
N = 1024*4

Nf = 3
Nms = 256
Tms = 100.0


class TestSignals(TestCase):
    def _signal(self, ds, dw, real=True):
        A0 = 0.5 + 0.5j
        A1 = 0.2 + 0.2j

        #Input signal
        u_in = A0*np.exp(1j*(w0+dw)*ds.t) + A1*np.exp(1j*(2*w0+dw)*ds.t)

        if real:    u_in = np.real(u_in)
        
        return A0,A1,u_in

    def T_multiscale(self, real=True):
        dsi = DiscreteSignal(N, T)
        msi = MultiscaleSignal(Nms, Tms, Nf, w0, real=real)

        dw = msi.omega[0,4]
        A0,A1,u_in = self._signal(dsi, dw, real=real)

        #Input signal for multiscale signal
        vm_in = np.zeros(msi.shape, np.complex_)
        vm_in[1] = A0*np.exp(1j*dw*msi.t[1])
        vm_in[2] = A1*np.exp(1j*dw*msi.t[2])

        #Check signals
        signal_difference(dsi, u_in, msi, vm_in, 'M'+'R'*real)

        #Check power
        assert_almost_equal(dsi.power(u_in), msi.power(vm_in))

    def T_periodic(self, real=True):
        dsi = DiscreteSignal(N, T)
        psi = PeriodicSignal(Nf, w0, real=real)

        A0,A1,u_in = self._signal(dsi, 0, real=real)

        #Input signal for periodic signal
        v0_in = np.zeros(psi.shape, np.complex_)
        v0_in[1] = A0
        v0_in[2] = A1

        #Check signals
        signal_difference(dsi, u_in, psi, v0_in, 'P'+'R'*real)

        #Check power
        assert_almost_equal(dsi.power(u_in), psi.power(v0_in))

    def test_real(self):
        self.T_periodic(real=True)
        self.T_multiscale(real=True)
        pass
        
    def test_complex(self):
        self.T_periodic(real=False)
        self.T_multiscale(real=False)
        pass


if __name__=="__main__":
    unittest.main()
    
