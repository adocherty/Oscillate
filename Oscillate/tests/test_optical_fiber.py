# encoding: utf-8


"""
Test the optical fiber propagation

 * Test full NLS solver
   - Using soliton solution

 * Test multiscale NLS solver

"""

import sys
sys.path.append("../..")

from numpy.testing import *
import unittest

import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi, linalg
from scipy import fftpack as fftp

from Optolib import *
from Optolib.opticalfiber import *
from Optolib.discrete import *
from Optolib.components import *
from Optolib.nlsesolvers import *


def assert_signals_almost_equal(ds1, v1, ds2, v2):
    signal1 = ds2.as_signal(ds1, v2)

    err =  linalg.norm(v1-signal1)/linalg.norm(v1)
    assert_almost_equal(err, 0)

def sech(x):
    return 1.0/np.cosh(x)

T = 256.0
N = 1024*16
Tms = T
Nms = 1024
Nf = 6
w0 = 2*pi*5.0

ds = DiscreteSignal(N, T)
ms = MultiscaleSignal(Nms, Tms, Nf, w0)

# Fiber parameters
An = 0
D2 = -1.
Gn = 1.
Ln = 5.0

# Direct solvers
ssf2 = SSF2RK2(An, 0.5*D2*ds.omega**2, Gn)
ssf24 = SSF2RK4(An, 0.5*D2*ds.omega**2, Gn)
ssf2e = SSF2Exact(An, 0.5*D2*ds.omega**2, Gn)

ssf24.set_numeric_parameters(adaptivestep=None, stepsize=0.001)
ssf2e.set_numeric_parameters(adaptivestep=None, stepsize=0.001)

#ssf2.set_numeric_parameters(adaptivestep=10, stepsize=0.01)
#ssf24.set_numeric_parameters(adaptivestep=10, stepsize=0.01)
#ssf2e.set_numeric_parameters(adaptivestep=10, stepsize=0.01)

# Multiscale solvers
ssf2ms = MultiscaleSSF2(An, 0.5*D2*ms.omega**2, Gn)
ssf24ms = MultiscaleSSF24(An, 0.5*D2*ms.omega**2, Gn)


class TestDirectPropagation(TestCase):
    def _signal(self):
        A = 1.0
        C = w0
        T0 =  np.mod(0.5*T + C*Ln, T)
        # Analytic solution
        uz_ana = A*sech(A*(ds.t - T0)) * np.exp(1j*C*ds.t) * np.exp(-0.5j*(C**2-A**2)*Ln)

        # Initial condition
        u0 = A*sech(A*(ds.t-0.5*T))*np.exp(1j*C*ds.t)
        return u0, uz_ana

    def _solve(self, S):
        S.set_numeric_parameters(adaptivestep=1)
        u0, uz_ana = self._signal()
        uz = S.solve(ds, u0, Ln)
        print "Err adaptive step:", np.abs(uz-uz_ana).max()
        #assert_array_almost_equal(uz, uz_ana,decimal=5)

        S.set_numeric_parameters(adaptivestep=None, stepsize=0.001)
        u0, uz_ana = self._signal()
        uz = S.solve(ds, u0, Ln)
        print "Err fixed step:", np.abs(uz-uz_ana).max()
        #assert_array_almost_equal(uz, uz_ana,decimal=5)

#        pl.plot(ds.t, np.abs(uz_ana),'r-')
#        pl.plot(ds.t, np.abs(uz),'k--')
#        pl.semilogy()

    def test_solvers(self):
        self._solve(ssf2)
        self._solve(ssf24)
        self._solve(ssf2e)

class TestMultiscalePropagation(TestCase):
    def _signal(self):
        A1 = 1.0
        A2 = 1.0

        # Initial condition
        u0 = A1*sech(A1*(ds.t-0.5*T)) + 0j
        u0 += A2*sech(A2*(ds.t-0.2*T))*np.exp(1j*w0*ds.t)

        u0_ms = np.zeros(ms.shape, np.complex_)
        u0_ms[0] = A1*sech(A1*(ms.t[0]-0.5*Tms))
        u0_ms[1] = A2*sech(A2*(ms.t[0]-0.2*Tms))
        return u0, u0_ms

    def _solve(self, S):

#        S.set_numeric_parameters(adaptivestep=1)
        S.set_numeric_parameters(adaptivestep=None, stepsize=0.0005)
        ssf2e.set_numeric_parameters(adaptivestep=1)

        u0, u0_ms = self._signal()
        uz = ssf2e.solve(ds, u0, Ln)
        uz_ms = S.solve(ms, u0_ms, Ln)

        sig1 = ms.as_signal(ds, uz_ms)
        err =  linalg.norm(uz-sig1)/linalg.norm(uz)
        print "Error:", err
        #assert_signals_almost_equal(ds,uz,ms,uz_ms)

        #pl.plot(ds.t, np.abs(uz),'k-')
        #pl.plot(ds.t, np.abs(sig1),'g--')
        #pl.plot(ds.t, np.abs(uz-sig1),'r--')
        pl.semilogy()

    def _test_solvers(self):
        self._solve(ssf2ms)
        #self._solve(ssf24ms)


if __name__=="__main__":
    unittest.main()
