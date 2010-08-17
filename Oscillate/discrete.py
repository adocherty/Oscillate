# _*_ coding=utf-8 _*_
#αβγδεζηθικλμνξοπρσςτυφχψω ϕϚϑ ‘’	“”  ·
#ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
#ℋℌℍℎℏℐℑℒℓℕ№℗℘ℙℚℛℜℝℤ℥ℨÅℬℰℱℳℼℽ∇
#¼½¾⅓ ⅔⅕⅖⅗⅘ ⅙⅚ ⅛

"""
Basic Signal classes for easy signal handling

Changes:
 * Delayed creation of vectors (t, f, ...) until they are requested

"""

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi
from scipy import random, signal, linalg
from scipy import fftpack as fftp

class Signal(object):
    pass

class DiscreteSignal(Signal):
    """
    A discretization class for a complex signal
    
    Parameters:
    -----------
    N: int
        Number of points in the timbase
    T: float
        The total size of the timebase
    """
    def __init__(self, N, T, defer=False):
        #Length of signal and timebase
        self._update(N, T)

    @property
    def t(self):
        """
        The time base of the signal
        """
        if not hasattr(self,'_t'):
            self._t = np.arange(0,self.N, dtype=np.float64)
            self._t *= self.dt
        return self._t

    @property
    def omega(self):
        """
        The Fourier angular frequency base of the signal
        """
        if not hasattr(self,'_omega'):
            self._omega = fftp.fftfreq(self.N,self.dt)  #Frequency base
            self._omega *= 2*pi
        return self._omega

    @property
    def f(self):
        """
        The Fourier frequency base of the signal
        """
        if not hasattr(self,'_f'):
            self._f = fftp.fftfreq(self.N,self.dt)  #Frequency base
        return self._f

    @property
    def t_onesided(self):
        """
        The time base of the signal
        """
        if not hasattr(self,'_t1s'):
            self._t1s = np.arange(0,self.N, dtype=np.float64)
            self._t1s *= self.dt
        return self._t1s

    @property
    def f_onesided(self):
        if not hasattr(self,'_f1s'):
            self._f1s = np.arange(0,self.N, dtype=np.float64)
            self._f1s /= self.T
        return self._f1s

    @property
    def omega_onesided(self):
        if not hasattr(self,'_omega1s'):
            self._omega1s = np.arange(0,self.N, dtype=np.float64)
            self._omega1s *= self.domega
        return self._omega1s


    def _update(self, N, T):
        self.N,self.T = N,T

        # Quantities derived from the input variables
        #Time discretization
        self.dt = np.float(self.T)/self.N
        
        #Frequency discretization
        self.df = 1.0/self.T
        self.domega = 2*pi/self.T

        #Vector shape
        self.shape = (self.N,)
        
        #Clear all calculated quantities
        self._clean_up()

    def _clean_up(self):
        # Clear all cached items
        for item in ['t','f','omega','t1s','f1s','omega1s']:
            try:
                delattr(self, '_'+item)
            except:
                pass

    def power(self, x, dc=1.0):
        """
        Return the average power of the signal
        P_av = 1/T \int_0^T |x|^2 dt = 1/N \sum_{n=0}^N |f_n|^2
        
        Note: This only corresponds to the periodic and multiscale
        average power if the signal contains an integer number of periods
        """
        # Calculate power in Fourier domain to isolate dc component
        xhat = self.fft(x)
        P = linalg.norm(xhat[1:])**2/self.N**2 + dc*np.abs(xhat[0])**2/self.N**2
            
        return P
        
    def fft(self, x, *args, **kwargs):
        return fftp.fft(x, *args, **kwargs)

    def ifft(self, x, *args, **kwargs):
        return fftp.ifft(x, *args, **kwargs)

    def fft_window(self, x, *args, **kwargs):
        return fftp.fft(x*self.t_window, *args, **kwargs)

    def info(self):
        print "Max frequency: %.5g Hz" % (1.0/self.dt/2)
        print "Frequency resolution: %.5g Hz" % (1.0/self.T)

        print "Max ang freq: %.5g rad/s" % (pi/self.dt)
        print "Ang freq resolution: %.5g rad/s" % (2*pi/self.T)
        
        print "Max time: %.5g s" % (self.T)
        print "Time resolution: %.5g s" % (self.dt)


class MultiscaleSignal(Signal):
    """
    A discretization class for a multiscale signal defined with
    fast and slow timescales:
    
    S(t) = 

    Parameters:
    -----------
    Nt: int
        Number of points in the timbase
    T: float
        The total size of the timebase
    Nms: int
        The number of multiscale signals
    w0: float
        The frequency spacing of the signals
    epsilon: float, optional
        The overall signal coefficient
    real: {True, False}, optional
        Flag for complex/real signals
    """
    def __init__(self, Nt, T, Nms, w0=1.0, real=False, epsilon=1.0):
        #Length of signal and timebase
        self.w0 = w0
        self.Nt, self.Nms = Nt,Nms
        self.T = T

        #We store the epsilon - the overall coefficient of the signal
        self.epsilon = epsilon

        #If the signal is real then only the positive frequencies are used
        self._real = real

        #Update all internal variables
        self._update_()

    @property
    def real(self):
        """
        Determines if the signal is real or complex.
        """
        return self._real

    @real.setter
    def real(self, value):
        self._real = value
        self._update_()

    def copy(self):
        ns = MultiscaleSignal(self.Nt, self.T, self.Nms, self.w0, self.real)
        return ns

    def _update_(self):
        #The multiscale frequency base
        if not self.real:
            self.shape = (2*self.Nms-1,self.Nt)
            self.ms = np.append(np.arange(0,self.Nms), np.arange(-self.Nms+1,0))
        else:
            self.shape = (self.Nms,self.Nt)
            self.ms = np.arange(0,self.Nms)

        #The slow-time slices
        self.dt = dt = self.T/self.Nt
        self.t = np.zeros(self.shape)
        
        #Time base
        self.t[:] = np.arange(0,self.T,dt)
        
        #Frequency base
        self.omega = 2*pi*fftp.fftfreq(self.Nt,dt)[np.newaxis,:] \
                     + self.w0*self.ms[:,np.newaxis]
        self.f = self.omega/2/pi
        self.domega = 2*pi/self.T

        #Two sided Hamming window
        self.t_window = 0.54-0.46*np.cos(2*pi*np.arange(self.Nt)/(self.Nt-1))

        #One sided bases
        self.t_ = np.arange(0,self.T,dt)   #Time base
        self.f_ = np.arange(self.Nms)/self.T
        self.omega_ = 2*pi*self.f_
    
    def convert_signal(self, ms_in, v_in):
        """
        Converts the discrete signal from real or complex
        to the current type of this discrete class
        """
        v_out = np.zeros(self.shape)
        
        #Convert from real to complex:
        if ms_in.real and not self.real:
            v_out[:ms_in.Nms] = v_in
            v_out[-ms_in.Nms:] = np.conj(v_in[::-1])
        elif not ms_in.real and self.real:
            v_out[:] = 0.5*v_in[:self.Nms]
            v_out[1:] += 0.5*v_in[-1:-self.Nms:-1]
        else:
            v_out[:] = v_in
        return v_out

    def take_part(self, x, N, ii=0):
        """
        Take a subsignal of the given signal
        
        Parameters
        ==========
        N : int
            Number of subsignals within signal
            (Nt must be divisible by N)
        ii : int
            Position of subsignal in signal (0 .. Nt//Nss)
        
        Returns
        =======
        ms : Signal
            multiscale signal for subsignal
        ssig : ndarray
            subsignal
        """
        Nt1 = self.Nt//N
        T1 = self.T/N
               
        ms1 = self.__class__(Nt1, T1, self.Nms, self.w0, self.real)
        ssig = x[..., ii*Nt1:(ii+1)*Nt1]
        return ms1, ssig        

    def store_part(self, x, ssig, N, ii=0):
        """
        Store a subsignal of the given signal
        
        Parameters
        ==========
        N : int
            Number of subsignals within signal
            (Nt must be divisible by N)
        ii : int
            Position of subsignal in signal (0 .. Nt//Nss)
        
        Returns
        =======
        ms : Signal
            multiscale signal for subsignal
        ssig : ndarray
            subsignal
        """
        Nt1 = self.Nt//N
        x[..., ii*Nt1:(ii+1)*Nt1] = ssig
        return self, x

    def resample(self, ds, v_in):
        """
        Resample signal
        """
        
        #Resample signal to other discrete class
        # 1) Trim signal so they are the same time
        Nt_ = int(ds.T/self.T*self.Nt)
        
        # 2) If there is only one sample then we
        #    want it to be duplicated to all ouput
        #    samples - signal.resample doesn't do
        #    this
        if Nt_==1:
            v_resample = np.ones(ds.N)*v_in[0]
        
        # If this occurs, we have a problem!
        elif Nt_<1:
            raise RuntimeError, "Resampling error"

        # Fourier resample to the new nodes
        else:
            v_resample = signal.resample(v_in[:Nt_], ds.N)
        
        return v_resample

    def as_signal(self, ds, amt):
        """
        Construct a signal in the time domain
        with a discrete class
        
        Parameters:
        -----------
        ds:     discrete signal with time domain
        ams:    periodic signal coefficients
        """
        if self.real:
            yt = np.zeros(ds.N, dtype=np.float_)
            for ii in range(self.shape[0]):
                aT = self.resample(ds,amt[ii])
                yt += np.real(aT*np.exp(1j*self.omega[ii,0]*ds.t))
        else:
            yt = np.zeros(ds.N, dtype=np.complex_)
            for ii in range(self.shape[0]):
                aT = self.resample(ds,amt[ii])
                yt += aT*np.exp(1j*self.omega[ii,0]*ds.t)

        return self.epsilon*yt

    def power(self, x, dc=1.0):
        """
        Returns the average power in signal represented by the given
        multiscale signal
        """
        xhat = self.fft(x)/self.Nt
        if self.real:
            P = 0.5*linalg.norm(xhat[1:].flatten())**2 + dc*linalg.norm(xhat[0].flatten())**2
        else:
            P = linalg.norm(xhat[1:].flatten())**2 + dc*linalg.norm(xhat[0].flatten())**2

        return P


    def fft(self, x, *args, **kwargs):
        """
        The FFT of the slow-time functions alone, not the entire signal
        """
        kwargs['axis'] = -1
        return fftp.fft(x, *args, **kwargs)

    def ifft(self, x, *args, **kwargs):
        """
        The IFFT of the slow-time functions alone, not the entire signal
        """
        kwargs['axis'] = -1
        return fftp.ifft(x, *args, **kwargs)

    def fft_window(self, x, *args, **kwargs):
        kwargs['axis'] = -1
        return fftp.fft(x*self.t_window, *args, **kwargs)

    def info(self):
        print "Max frequency: %.5g Hz" % (1.0/self.dt/2)
        print "Frequency resolution: %.5g Hz" % (1.0/self.T)

        print "Max ang freq: %.5g rad/s" % (pi/self.dt)
        print "Ang freq resolution: %.5g rad/s" % (2*pi/self.T)
        
        print "Max time: %.5g s" % (self.T)
        print "Time resolution: %.5g s" % (self.dt)

        print "Number of harmonics of fast timescale: %d" % (self.Nms)


class PeriodicSignal(Signal):
    """
    A discretization class for a periodic signal
    defined by Fourier series coefficients

    There are 2Nf-1 Fourier coefficients for complex
    signals and Nf coefficients for real signals.
    
    Parameters:
    -----------
    Nf: int
        Number of positive Fourier components
    w0: float
        The signal's angular frequency
    real: bool, optional
        Flag, indicates if the signal is real or complex
    """
    def __init__(self, Nf, w0, real=False):
        self.Nf,self.w0 = Nf,w0
        self._real = real
        self._update_()

    @property
    def real(self):
        """
        Determines if the signal is real or complex.
        """
        return self._real

    @real.setter
    def real(self, value):
        self._real = value
        self._update_()
        
    def _update_(self):
        if not self.real:
            self.N = 2*self.Nf-1
            self.ms = np.append(np.arange(0,self.Nf), np.arange(-self.Nf+1,0))
        else:
            self.N = self.Nf
            self.ms = np.arange(0,self.N)

        self.omega = self.ms*self.w0
        self.f = self.omega/2/pi

        #One sided spectra
        self.omega_ = np.arange(0,self.N)*self.w0
        self.f_ = self.omega_/2/pi

        #Vector Shape
        self.shape = (self.N,)

    def copy(self):
        """
        Returns a periodic signal object with the same
        parameters as the current one
        """
        ns = PeriodicSignal(self.Nf, self.w0, self.real)
        return ns
        
    def as_signal(self, ds, ams):
        """
        Construct signal in time domain
        
        Parameters:
        -----------
        ds: DiscreteSignal
            discrete signal with time domain
        ams: ndarray
            periodic signal coefficients
        """
        
        if self.real:
            yt = np.zeros(ds.N, dtype=np.float_)
            for ii in xrange(0,self.N):
                yt += np.real(ams[ii]*np.exp(1j*self.omega[ii]*ds.t))

        else:
            yt = np.zeros(ds.N, dtype=np.complex_)
            for ii in xrange(0,self.N):
                yt += ams[ii]*np.exp(1j*self.omega[ii]*ds.t)

        return yt

    def power(self, ams, dc=1.0):
        """
        Returns the average power in a signal represented by
        the given Fourier series coefficents
        """
        if self.real:
            P = 0.5*linalg.norm(ams[1:])**2 + dc*np.abs(ams[0])**2
        else:
            P = linalg.norm(ams[1:])**2 + dc*np.abs(ams[0])**2

        return P

    def fft(self, x, *args, **kwargs):
        """
        This is defined to mean the FT of the data around each harmonic,
        namely we can ignore this        
        """
        return x

    def ifft(self, x, *args, **kwargs):
        """
        This is defined to mean the FT of the data around each harmonic,
        namely we can ignore this        
        """
        return x


    def set_coeffs(self, ams):
        self.ams = ams

    def get_coeffs(self):
        return self.ams
