# _*_ coding=utf-8 _*_
#αβγδεζηθικλμνξοπρσςτυφχψω ϕϚϑ ‘’	“”  ·
#ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
#ℋℌℍℎℏℐℑℒℓℕ№℗℘ℙℚℛℜℝℤ℥ℨÅℬℰℱℳℼℽ∇
#¼½¾⅓ ⅔⅕⅖⅗⅘ ⅙⅚ ⅛

"""
Basic Stochastic Routines

"""

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi
from scipy import random, signal
from scipy import fftpack as fftp


class StochasticSimulator(object):
    def __init__(self, discrete, signal, system, detector):
        self.discrete = discrete
        self.signal = signal
        self.system = system
        self.detector = detector
    
    def run(self, Ne):
        ds = self.discrete
        Nmeas = self.detector.Nmeas
        
        #Build discrete ensemble of processes
        self.xe = np.zeros((Nmeas,Ne,ds.N))
        
        for ii in range(Ne):
            #Signal
            y_out = self.signal(ds)

            #System list
            for S in self.system:
                y_out = S(ds, y_out)

            #Detector
            y_meas, t_meas = self.detector(ds, y_out)

            self.xe[:,ii] = y_meas
            
    def analyze(self, xe):
        ds = self.discrete

        # Sampled spectral density
        Vx = ds.fft_window(xe)

        #Estimate of expectation
        Ex = np.mean(xe, axis=0)

        #Other moments
        Ex2 = np.mean(np.abs(xe)**2, axis=0)

        #Two-sided sampled spectrum and expectation
        Xe = ds.fft(xe, axis=1)
        X = np.mean(Xe, axis=0)
        Stilde = np.abs(Xe)**2/ds.N
        S = np.mean(Stilde, axis=0)*ds.dt

        return X, S
    
    def plot(self, meas=0, wmax=None, dbchz=False, style=''):
        ds = self.discrete
        
        X, S = self.analyze(self.xe[meas])

        if wmax:
            Nmax = int(wmax/ds.domega)
        else:
            Nmax = ds.N//2

        if dbchz:
            pl.plot(ds.f_[:Nmax], 10*np.log10(S[:Nmax]), style, label='PSD')
        else:
            pl.plot(ds.f_[:Nmax], S[:Nmax], style, label='PSD')

        #plot_shifted(ds.f, S, style, label='PSD')

