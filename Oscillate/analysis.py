# encoding: utf-8

"""
Analysis

"""


import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi
from scipy import random, signal, stats
from scipy import fftpack as fftp

from components import *
from discrete import *

class Analysis(object):
    """
    """
    def __init__(self, ds=None, vsignal=None):
        self.ds = ds
        self.x = vsignal

    def set_signal(self, ds, vsignal):
        self.ds = ds
        self.x = vsignal

    def get_findex_range(self, fstart=None, fend=None):
        # Select data range
        Nstart = None if fstart is None else int(fstart/self.ds.df)
        Nend = None if fstart is None else int(fend/self.ds.df)
        return slice(Nstart,Nend)

    def spectrum(self, fstart=None, fend=None):
        # Axis
        f = self.ds.f

        # Data
        xhat = self.ds.fft(self.x)
        xhat /= self.ds.N

        # Slice raw data
        sl = self.get_findex_range(fstart,fend)
        fret = f[sl]
        xret = xhat[sl]

        return fret, xret

    def phase_noise(self, dbc=False):
        phi_m = np.angle(self.x)
        #phi_m = np.unwrap(phi_m)
        
        # Remove mean and 1st moment - why?
        #a1,a0 = np.polyfit(self.ds.t, phi_m, 1)
        #phi_m -= a0 + a1*self.ds.t
        return phi_m

    def amplitude_noise(self, dbc=False):
        a_m = np.abs(self.x)
        a_m /= np.mean(a_m)
        a_m -= 1.0
        return a_m

    def phase_noise_spectrum(self, fstart=None, fend=None, dbc=False):
        # Axis
        f = self.ds.f

        # Data
        self.phihat = self.phase_noise()
        self.phihat = self.ds.fft(self.phihat)
        self.phihat /= self.ds.N

        # Slice raw data
        sl = self.get_findex_range(fstart,fend)
        fret = f[sl]
        phiret = self.phihat[sl]

        return fret, phiret

    def amplitude_noise_spectrum(self, fstart=None, fend=None, dbc=False):
        # Axis
        f = self.ds.f

        # Data
        ahat = self.amplitude_noise()
        ahat = self.ds.fft(ahat)
        ahat /= self.ds.N

        sl = self.get_findex_range(fstart,fend)
        fret = f[sl]
        aret = ahat[sl]

        return fret, aret

    def deviation(self):
        """
        Return standard deviation of the noise of the entire signal
        This is of both phase and amplitude noise
        """
        return np.std(self.x)

    def signaltonoise(self):
        """
        The signal to noise ratio
        """
        return np.mean(np.abs(self.x))/np.std(self.x)
        
    def noisetosignal(self):
        """
        The noise to signal ratio
        """
        return np.std(self.x)/np.mean(np.abs(self.x))


class EnsembleAnalysis(object):

    def add_signal(self, ds, vsignal):
        """
        Add signal to ensemble
        """
        if ds is not None:
            self.ds = ds
            self.ensemble.append(vsignal)
    
    def ensemble_average(self, x):
        return np.mean(x, axis=0)

class AnalysisFile(object):
    """
    Use memmap to analyse a large set of data
    and not store it all in memory
    """
    pass



class LoopStorage(object):
    def __init__(self, ds, Nstore, Nsig):
        self._storage_ = np.zeros((Nstore,Nsig), dtype=np.complex_)
        self.current_location = 0
        self.current_sample = 0.
        self.ds = ds

    def add_sample(self, sample_number=None, reset=True):
        if sample_number is None:
            self.current_sample += 1.
        else:
            self.current_sample = sample_number

        if reset:
            self.current_location = 0

    def store(self, ds, vsig, ii=None):
        """
        """
        #assert ds.shape == self.shape, "Signal shape must be the same as the initial signal"
        if ii is None:
            ii = self.current_location
        
        #Store at postion
        self._storage_[ii] = vsig
        
        #Increment position
        self.current_location = ii+1

    def store_running_average(self, ds, vsig, ii=None):
        """
        """
        #assert ds.shape == self.shape, "Signal shape must be the same as the initial signal"
        if ii is None:
            ii = self.current_location
        
        #Store at postion
        self._storage_[ii] *= self.current_sample
        self._storage_[ii] += vsig
        self._storage_[ii] /= self.current_sample + 1.
        
        #Increment position
        self.current_location = ii+1

    def time_concatenated_signal(self, n=None, samples=None, start=None, end=None):
        """
        Return signal from concatenated stored timedomain signals
        """
        Nstore = self._storage_.shape[0]

        # Break into sample signals
        if n is not None and n<samples:
            chunk_size = Nstore//samples
            start = n*chunk_size
            end = (n+1)*chunk_size

        #vcc = self._storage_[start:end]
        vcc = self._storage_

        #Create concatenated signal
        dscc = DiscreteSignal(np.prod(vcc.shape), self.ds.T*vcc.shape[0])
        
        return dscc, vcc.flatten()
        
        

