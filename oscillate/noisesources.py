# encoding: utf-8
"""
Noise sources for optical and electronic components

"""
import timer

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi
from scipy import random, signal
from scipy import fftpack as fftp

from components import SourceComponent,Component

try:
    import noisegeneration as ng
except:
    pass

def discrete_white_noise(N,dt,Q):
    #Discrete noise variance
    Qd = Q/dt

    # Generate some white noise with σ=1
    w = sp.random.normal(0, np.sqrt(Qd), N)
    return w

class WhiteNoiseSource(SourceComponent):
    def __init__(self, nd=0):
        self.nd = np.sqrt(nd/2)
        self.set_source_channels()
        
    def set_source_channels(self, channels=None):
        self.source_channels = channels

    def _generate_noise_(self, N, dt):
        # Shared noise function:
        # Generate noise with a vector shape N
        xt = np.zeros(N, dtype=np.complex_)
        
        if self.source_channels is None:
            scs = range(N[0])
        else:
            scs = np.atleast_1d(self.source_channels)

        for ii in scs:
            n1 = discrete_white_noise(N[1], dt, 1.0)
            n2 = discrete_white_noise(N[1], dt, 1.0)
            xt[ii] += self.nd*(n1 + 1j*n2)
        return xt

    def direct(self, ds):
        """
        Source for simulations with full time dependence
        """
        return self._generate_noise_(ds.shape,ds.dt)

    def order1(self, ps):
        """
        Source for O(1) periodic input
        """
        return 0

    def ordere(self, ms):
        """
        Source for O(ε) slow-time input
        """
        return self._generate_noise_(ms.shape,ms.dt)

    def multiscale(self, ms):
        """
        Noise source for a multiscale signal
        """
        return self._generate_noise_(ms.shape,ms.dt)



class FlickerNoiseSource(SourceComponent):
    pass


class DeltaSource(SourceComponent):
    def __init__(self, b0=0, b1=0, b2=0, angle=0):
        self.nd0 = np.sqrt(b0/2)*np.exp(1j*angle)
        self.nd1 = np.sqrt(b1/2)*np.exp(1j*angle)
        self.nd2 = np.sqrt(b2/2)*np.exp(1j*angle)
        
        self.set_source_channels()
        
    def set_source_channels(self, channels=None):
        self.source_channels = channels

    @timer.time_function(prefix="noisesource:")
    def _generate_noise_(self, N, dt):
        # Shared noise function: Generate noise
        # with a vector shape N, 
        xt = np.zeros(N, dtype=np.complex_)
        
        if self.source_channels is None:
            scs = range(N[0])
        else:
            scs = np.atleast_1d(self.source_channels)

        deltat = fftp.ifft(np.ones(N[1]))/dt

        for ii in scs:
            if self.nd0>0:
                xt[ii] += self.nd0*deltat
            if self.nd1>0:
                xt[ii] += self.nd1*deltat1
            if self.nd2>0:
                xt[ii] += self.nd1*deltat2

        return xt

    def direct(self, ds):
        """
        Source for simulations with full time dependence
        """
        return self._generate_noise_(ds.shape,ds.dt)

    def order1(self, ps):
        """
        Source for O(1) periodic input
        """
        return 0

    def ordere(self, ms):
        """
        Source for O(ε) slow-time input
        """
        return self._generate_noise_(ms.shape,ms.dt)

    def multiscale(self, ms):
        """
        Noise source for a multiscale signal
        """
        return self._generate_noise_(ms.shape,ms.dt)

