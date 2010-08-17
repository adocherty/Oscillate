# _*_ coding=utf-8 _*_
#αβγδεζηθικλμνξοπρσςτυφχψω ϕϚϑ ‘’	“”  ·
#ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
#ℋℌℍℎℏℐℑℒℓℕ№℗℘ℙℚℛℜℝℤ℥ℨÅℬℰℱℳℼℽ∇
#¼½¾⅓ ⅔⅕⅖⅗⅘ ⅙⅚ ⅛

"""
Detectors and receivers
-----------------------

Currently there is only a direct detector which
must be paired with a proper filter:
 * DirectDetector()

And a choice of receivers, all taking the carrier
frequency as an argument and returning as measurements
the amplitude and phase noise
 * QuadratureReceiver(w0)
 * AsymptoticReceiver(w0)

"""

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi
from scipy import random, signal
from scipy import fftpack as fftp

from components import Component, TransferComponent

try:
    import specializedconvolve as sc
except:
    print "Warning: detection not available without compilation"
    #import slowconvolve as sc

# Debugging/profiling
import timer

def db_to_coeff(iloss):
    return 10**(0.1*iloss)

def ps_flip(ams, axis=0):
    return np.append(ams[0], ams[:0:-1], axis=axis)

def ps_shift(ams, axis=0):
    N = ams.shape[axis]//2
    return np.append(ams[-N:],ams[:N+1], axis=axis)

#
#
# Measurement detectors
#
#
class PhaseNoiseMeasurement(Component):
    """
    Returns the phase noise of the signal
    """
    def __init__(self, w0=None):
        self.w0 = w0

    def direct(self, ds, vt):
        t, omega = ds.t, ds.omega
        if self.w0:
            w0 = self.w0
        else:
            raise RuntimeError, "Oscillation frequency required"

        # Quadrature Heterodyne demodulation
        vt_dc_c = vt*np.cos(w0*t)
        vt_dc_s = vt*np.cos(w0*t + pi/2)

        # Filter
        ffilt = np.float_(np.absolute(omega) < w0)
        vt_dc_c = ds.ifft(ds.fft(vt_dc_c)*ffilt)
        vt_dc_s = ds.ifft(ds.fft(vt_dc_s)*ffilt)

        # Measure phase noise
        phi_out = np.unwrap(np.arctan2(vt_dc_s.real,vt_dc_c.real))
        a_out = np.sqrt(vt_dc_s**2 + vt_dc_c**2)
        a_out = a_out/np.mean(a_out) - 1.0
        
        return phi_out

    def multiscale(self, ds, vm):
        phi_m = np.angle(vm)
        return phi_m

    def order1(self, ds, v0):
        return 0

    def ordere(self, ds, vm):
        phi_m = np.angle(vm)
        return phi_m



class AmplitudeNoiseMeasurement(Component):
    def __init__(self, w0=None):
        self.w0 = w0

    def direct(self, ds, vt):
        t, omega = ds.t, ds.omega
        if self.w0:
            w0 = self.w0
        else:
            raise RuntimeError, "Oscillation frequency required"

        # Quadrature Heterodyne demodulation
        vt_dc_c = vt*np.cos(w0*t)
        vt_dc_s = vt*np.cos(w0*t + pi/2)

        # Filter
        ffilt = np.float_(np.absolute(omega) < w0)
        vt_dc_c = ds.ifft(ds.fft(vt_dc_c)*ffilt)
        vt_dc_s = ds.ifft(ds.fft(vt_dc_s)*ffilt)

        # Measure amplitude noise
        a_out = np.sqrt(vt_dc_s**2 + vt_dc_c**2)
        a_out = a_out/np.mean(a_out) - 1.0
        
        return a_out

    def multiscale(self, ds, vm):
        a_m = np.abs(vm)
        a_m = a_m/np.mean(a_m) - 1.0
        return phi_m

    def order1(self, ds, v0):
        return 0

    def ordere(self, ds, vm):
        a_m = np.abs(vm)
        return a_m


class AsymptoticReceiver(Component):
    Nmeas = 2
    def __init__(self, w0):
        self.w0 = w0
    def direct(self, ds, vt_out):
        t, omega = ds.t, ds.omega
        w0 = self.w0

        # Recover amplitude and phase
        filt = np.float_(np.absolute(ds.omega) < w0)

        y = vt_out
        at_r = ds.ifft(ds.fft(y*np.exp(1j*w0*t) + y*np.exp(-1j*w0*t)) * filt)
        pt_r = 1j*ds.ifft(ds.fft(y*np.exp(1j*w0*t) - y*np.exp(-1j*w0*t)) * filt)

        return [at_r, pt_r], t


class DirectDetector(TransferComponent):
    """
    Direct optical detection
    
    The output electrical voltage is proportional to the optical power
    
    V(t) = iloss rho R |E(t)|^2
    
    Where
    iloss: insertion loss
    rho:   responsivity
    R:     load impedence (Ω)
    """
    def __init__(self, rho=0.8, R=1.0, iloss=1.0):
        self.Vph = rho*R
        self.set_insertion_loss(iloss)

    def direct(self, ds, vt_in):
        """
        Direct equation with full time dependence
        """
        vt_det = self.eloss*self.Vph*np.abs(vt_in)**2
        return ds, vt_det

    @timer.time_function(prefix="det:")
    def multiscale(self, ms, vm_in):
        """
        Equation assuming O(1) periodic input
        
        Input is optical phasor, output is electrical phasor
        
        V_m(t) = Σ_k E_k(t) E^*_{k-m}(t)
        
        ms:     input discrete multiscale
        vm_in:  multiscale input
        """
        #Output signal will be real
        mso = ms.copy()
        mso.real = True

        #Calculate |v|^2 v by convolution
        absv2 = sc.df_convolve_to_real(vm_in,vm_in.conjugate())

        #Output - put into correct real representation
        vm_det = self.eloss*self.Vph*absv2
        vm_det[1:] *= 2.0

        return mso, vm_det

    @timer.time_function(prefix="det:")
    def order1(self, ps, v0_in):
        """
        Equation assuming O(1) periodic input
        
        Input is optical phasor, output is electrical phasor
        
        Vm = Σ_k E_k E^*_{k-m}
        
        ps:     input periodic signal
        vo_in:  O(1) input signal
        """
        #Output signal will be real
        pso = ps.copy()
        pso.real = True

        vma = ps_shift(v0_in)
        vmb = np.conj(ps_shift(v0_in))[::-1]
        v0_det = self.eloss*self.Vph*np.convolve(vma, vmb, mode='same')[-pso.N:]
        v0_det[1:] *= 2.0

        #Store the O(1) signal
        self.O1_solution = (pso,v0_in)
        self.O1_complete = True

        return pso, v0_det

    @timer.time_function(prefix="det:")
    def ordere(self, ms, v1_in):
        """
        Detector function assuming O(ε) input
        
        Input is optical phasor, output is electrical phasor
        
        With O(1) signal a_k
        and O(ε) sigal b_k(t)
        
        V_m = Σ_k a_k b^*_{k-m} + a^*_{k-m} b_m
        
        ps:     input periodic signal
        v0_in:  O(1) input signal

        ms:     input multiscale signal
        v1_in:  O(ε) input signal
        """
        #Retreive the O(1) solution
        assert self.O1_complete, "O(1) simulation must be complete before O(ε) can be run"
        (ps,v0_in) = self.O1_solution
        
        #Output signal will be real
        mso = ms.copy()
        mso.real = True

        v1_det = np.zeros(mso.shape, np.complex_)

        vma = ps_shift(v0_in)                   # a_k
        vmb = np.conj(ps_shift(v1_in))          # b^*_{-k}
        v1_det = self.Vph*_slow_convolve(vma,vmb,1)[::-1][:mso.Nms]

        vma = np.conj(ps_shift(v0_in))          # a^*_{-k}
        vmb = ps_shift(v1_in)                   # b_{-k}
        v1_det += self.eloss*self.Vph*_slow_convolve(vma,vmb,1)[::-1][:mso.Nms]

#        for ii in range(mso.Nt):
#            v1_det[:,ii] = self.Vph*signal.convolve(vma, vmb[:,ii], \
#                    mode='same', old_behavior=False)[-mso.Nms:]

#        for ii in range(mso.Nt):
#            v1_det[:,ii] += self.Vph*signal.convolve(vma, vmb[:,ii], \
#                    mode='same', old_behavior=False)[-mso.Nms:]

        v1_det[1:] *= 2.0
        return mso, v1_det


