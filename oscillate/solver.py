# _*_ coding=utf-8 _*_
#αβγδεζηθικλμνξοπρσςτυφχψω ϕϚϑ ‘’	“”  ·
#ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
#ℋℌℍℎℏℐℑℒℓℕ№℗℘ℙℚℛℜℝℤ℥ℨÅℬℰℱℳℼℽ∇√
#¼½¾⅓ ⅔⅕⅖⅗⅘ ⅙⅚ ⅛
#⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ 

"""
Library of optical fibers and propagation
-----------------------------------------

Currently there are optical fibers:
GenericFiber(beta, gamma, length)

CorningSMF()

"""

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi
from numpy import linalg as la
from scipy import random, signal, integrate
from scipy import fftpack as fftp
from scipy.interpolate import interp1d

# OEO Modules
from components import *
from nlsesolvers import *
from nlsesolvers_fast import *

from odesolve import *

import timer

#Speed of light
c = 2.998e8               # - m/s


# -----------   Propagation Solvers  -----------------

class LinearOpticalPropagation(LinearComponent):
    """
    Propagates an optical signal linearly through an optical component
    
    Arguments:
     * wl:  wavelength
     * fiber_list: a list of fibers to propagate the signal
       through
    """
    def __init__(self, fiber_list, wl=1.0):
        self.wl = wl
        self.fibers = fiber_list
        self.set_insertion_loss(1.0)

    def transfer_function(self, omega):
        fiber = self.fibers[0]
        
        alpha = fiber.alpha(self.wl)
        betaw = fiber.beta(self.wl, omega)
        L = fiber.length

        Hw = np.exp(-0.5*alpha*L)
        Hw *= np.exp(1j*betaw*L)
        return Hw


class NonlinearLinearOpticalPropagation(Component):
    """
    Propagates an optical signal linearly through a fiber
    with periodic boundary conditions.
    
    Arguments:
     * wl:  wavelength
     * fiber_list: a list of fibers to propagate the signal
       through
    """
    def __init__(self, fiber_list, wl=1.0):
        self.wl = wl
        self.fibers = fiber_list
        self.set_insertion_loss(1.0)

    def direct(self, ds, vt_in):
        """
        Direct equation with full time dependence
        """
        fiber = self.fibers[0]
        ssft = SSFTSolver(fiber.alpha(self.wl),
                fiber.beta(self.wl, ds.omega),
                fiber.gamma(self.wl),
                tol=1e-8)
        
        #Solve optical propagation over fiber length
        vt_out = ssft(vt_in, fiber.length)

        return ds, vt_out

    @timer.time_function(prefix="fi:")
    def order1(self, ps, vm_in):
        """
        Optical propagation of O(1) periodic input
        
        Here we take the input as an phasor in the optical domain
        and the output is returned as a phasor in the optical domain
        
        vm_out = [a₀, a₁, a₂ ... aᵥ, a₋ᵥ ... a₋₂, a₋₁, a₋₀]

        where the optical signal at optical carrier frequency wc is:
        E(t) = sum_{k=-v}^v a_k exp(j k w0 t) exp(j wc t)

        using the positive frequency convention (as in Rubiola)
        """

        from scipy.integrate import ode

  
        fiber = self.fibers[0]
        alpha = np.complex128(fiber.alpha(self.wl))
        gamma = np.complex128(fiber.gamma(self.wl))
        betaw = fiber.beta(self.wl, ps.omega).astype(np.complex128)

        r = ode(df_nls_order1)
        r.set_integrator('zvode', nsteps=1000, atol=1e-10, rtol=1e-10)
        r.set_initial_value(vm_in, 0)
        r.set_f_params(alpha,gamma,betaw)

        #Solve
        r.integrate(fiber.length)

        return ps, r.y

    def ordere(self, ds, vt_in):
        """
        Modulator response to O(ε) slow-time input
        linearized about an O(1) behaviour
        """
        
        return ds, vt_out


