# _*_ coding=utf-8 _*_
#αβγδεζηθικλμνξοπρσςτυφχψω ϕϚϑ ‘’	“”  ·
#ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
#ℋℌℍℎℏℐℑℒℓℕ№℗℘ℙℚℛℜℝℤ℥ℨÅℬℰℱℳℼℽ∇√
#¼½¾⅓ ⅔⅕⅖⅗⅘ ⅙⅚ ⅛

"""
Module containing implementations of amplifiers classes

SimpleSaturableAmplifier:   Simple saturable amplifier

"""

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi
from numpy import linalg as la
from scipy import fftpack as fftp

from components import Component, TransferComponent

# Debugging/profiling
import timer

class SimpleSaturableAmplifier(TransferComponent):
    """
    A simple saturable amplifier with gain
    
    g(P) = g_0 / (1 + P/P_0)

    where P is the electronic signal power
    
    Usage:
    ------
    SimpleSaturableAmplifier(g0, P0)
    g0: Small-signal gain
    P0: Saturation power
    
    """
    def __init__(self, g0, P0, iloss=1.0):
        self.set_gain(g0)
        self.P0 = P0
        self.set_insertion_loss(iloss)

    def gain(self, P):
        g = self.eloss*self.g/(1.0 + P/self.P0)
        return g

    def set_gain(self, g):
        self.g = g

    def direct(self, ds, vt_in):
        """
        Direct equation with full time dependence
        """
        #Calculate signal power without DC component
        P = ds.power(vt_in, dc=0)

        return ds, self.gain(P)*vt_in

    @timer.time_function(prefix="amp:")
    def multiscale(self, ms, vt_in):
        """
        Fully nonlinear multiscale equation in phasor representation
        """
        P = ms.power(vt_in, dc=0)
        return ms, self.gain(P)*vt_in

    def order1(self, ps, v0_in):
        """
        Equation assuming O(1) periodic input
        """
        P = ps.power(v0_in, dc=0)

        #Store needed information for O(ε) solution
        self.O1_complete = True
        self.O1_solution = (P,)

        return ps, self.gain(P)*v0_in

    def ordere(self, ms, v1_in):
        """
        Linearized response to O(ε) slow-time input
        """
        #Retreive the O(1) solution
        assert self.O1_complete, "O(1) simulation must be complete before O(ε) can be run"
        (P,) = self.O1_solution

        return ms, self.gain(P)*v1_in


class ParametricAmplifier(TransferComponent):
    """
    """
    def __init__(self, alpha, iloss=1.0):
        self.alpha = alpha
        self.set_insertion_loss(iloss)

    def direct(self, ds, vt_in):
        """
        Direct equation with full time dependence
        """
        #Calculate signal power without DC component
        P = ds.power(vt_in, dc=0)
        
        return ds, self.gain(P)*vt_in

    @timer.time_function(prefix="amp:")
    def multiscale(self, ms, vt_in):
        vhat = ms.fft(vt_in)/ms.Nt

        P1 = 0.5*la.norm(vhat[1,0].flatten())**2
        P2 = 0.5*la.norm(vhat[1,1:].flatten())**2
        
        self.eta2 = 1.0 + self.alpha
        if (P1+P2)>self.eta2**2*P2:
            self.eta1 = np.sqrt((P1+P2)/P1 - self.eta2**2*P2/P1)
        else:
            self.eta2 = self.eta1 = 1.0

        vhat[1,0] *= self.eta1
        vhat[1,1:] *= self.eta2

        vt_out = self.eloss*fftp.ifft(vhat*ms.Nt)
        return ms, vt_out

    def order1(self, ms, vt_in):
        return ms, vt_in


class NoiseTransfer(TransferComponent):
    """
    Component to mix amplitude and phase noise
    
    α_out = E_11 α_in + E_21 φ_in
    φ_out = E_12 α_in + E_22 φ_in
    
    """
    def __init__(self, E, iloss=1.0):
        self.E = E
        self.set_insertion_loss(iloss)

    def direct(self, ds, vt_in):
        return ds, vt_in

    def multiscale(self, ms, vt_in):
        v1_av = np.mean(vt_in[1])
        if np.abs(v1_av)==0:
            return ms, vt_in

        #Extract amp and phase noise
        v1_extract = (vt_in[1]/v1_av) - 1
        num_an = np.real(v1_extract)
        num_pn = np.imag(v1_extract)

        an = self.E[0,0]*num_an + self.E[0,1]*num_pn
        pn = self.E[1,0]*num_an + self.E[1,1]*num_pn

        #Reconstruct signal        
        v1_out = vt_in.copy()
        v1_out[1] = v1_av*((1+an) + 1j*pn)
        
        return ms, v1_out

    def order1(self, ms, vt_in):
        return ms, vt_in

