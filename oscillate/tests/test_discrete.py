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

from oscillate import *
from oscillate.opticalfiber import *
from oscillate.discrete import *
from oscillate.components import *
from oscillate.modulators import *
from oscillate.amplifiers import *
from oscillate.detectors import *
from oscillate.noisesources import *
from oscillate.analysis import *


w0 = 2*pi

T = 25.0
N = 10

t_ = np.asarray([0., 2.5, 5., 7.5, 10., 12.5, 15., 17.5, 20., 22.5])
f_ = np.asarray([0., 0.04, 0.08, 0.12, 0.16, -0.2 , -0.16, -0.12, -0.08, -0.04])
omega_ = f_*2*pi

t_os = np.asarray([0., 2.5, 5., 7.5, 10., 12.5, 15., 17.5, 20., 22.5])
f_os = np.asarray([0., 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36])
omega_os = f_os*2*pi

Nf = 3
Nms = 10
Tms = 250.0

class TestDiscrete(TestCase):
    def test_discrete(self, real=True):
        dsi = DiscreteSignal(N, T)

        assert_array_equal(dsi.omega, omega_)
        assert_array_equal(dsi.f, f_)
        assert_array_equal(dsi.t, t_)

        assert_array_equal(dsi.f_onesided, f_os)
        assert_array_almost_equal(dsi.omega_onesided, omega_os)
        assert_array_equal(dsi.t_onesided, t_os)

if __name__=="__main__":
    unittest.main()
