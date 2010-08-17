# encoding: utf-8


"""
Module of standard components

Base Classes:
------------
Component:          Base class for all components
Addition:           Simple addition of two signals
Mixer:              Mix two signals at different rations
TransferComponent:  Generic single input single output component
SourceComponent:    Generic source component
LinearComponent:    Generic linear component defined by transfer function

Linear Components:
-----------------
Amplifier:                  Linear amplifier
LorenzianFilter:            Lorenzian filter

RectangularBandpassFilter:  Exact rectangular filter
RectangularHighpassFilter:  Exact rectangular filter
RectangularLowpassFilter:   Exact rectangular filter

"""

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi
from numpy import linalg as la
from scipy import fftpack as fftp

# Debugging/profiling
import timer

class Loop(object):
    def __init__(self, components=[]):
        self.components = components
        self._debug_ = False
    
    def add_component(self, c, inputs=None, outputs=None):
        """
        Add a component to the sysytem
        
        Optionally, specifiy the input and output connections
        (Currently unused)
        """
        if inputs:
            c.connect_input(inputs)
        if outputs:
            c.connect_output(inputs)
        
        self.components.append(c)
        
    def order1(self, pss, v0_sig):

        for c in self.components:
            if isinstance(c, SourceComponent):
                v0_sig += c.order1(pss)
            
            elif isinstance(c, TransferComponent):
                pss,v0_sig = c.order1(pss, v0_sig)

        return pss,v0_sig

    def multiscale(self, mss, v_sig, sources=True):

        for c in self.components:
            if sources and isinstance(c, SourceComponent):
                v_sig += c.multiscale(mss)
            
            elif isinstance(c, TransferComponent):
                mss,v_sig = c.multiscale(mss, v_sig)
            
            if self._debug_:
                print "%s: %s" % (c.__class__.__name__, v_sig[:3,0])
                    
        return mss,v_sig

    def singlepass(self, ms, v_sig, sources=True):
        """
        Execute loop for signal passing though all components
        """
        return self.multiscale(ms, v_sig, sources)

    def calc_ss_gain(self, ps, eps=1e-6, nn=1):
        """
        Calculate small signal loop gain
        """
        v0_sig = np.zeros(ps.shape, np.complex_)
        vss_sig = np.copy(v0_sig)
        vss_sig[nn] += eps

        for c in self.components:
            if isinstance(c, TransferComponent):
                ps, v0_sig = c.multiscale(ps, v0_sig)
                ps, vss_sig = c.multiscale(ps, vss_sig)

        # Compatibility with ps and ms signals
        if len(ps.shape)>1:
            ssgain = np.abs(np.mean((v0_sig[nn]-vss_sig[nn])/eps))
        else:
            ssgain = np.abs((v0_sig[nn]-vss_sig[nn])/eps)
        return ssgain



class Component(object):
    """
    Base class for all components
    """
    def set_insertion_loss(self, iloss=1.0):
        self.iloss = iloss
        self.eloss = np.sqrt(iloss)

    def connect_input(self, cin):
        self.input = cin

    def connect_output(self, cout):
        self.output = cout

class Addition(Component):
    O1_complete = False
    def direct(self, ds, vt1, vt2):
        """
        Direct equation with full time dependence
        """
        return ds, vt1+vt2

    def order1(self, ps, vm, vm2):
        """
        Equation assuming O(1) periodic input
        """
        return ps, vm1+vm2

    def ordere(self, ms, vt1, vt2):
        """
        Linearized response to O(ε) slow-time input
        """
        return ms, vt1+vt2

    def multiscale(self, ms, vt1, vt2):
        """
        Fully nonlinear multiscale equation in phasor representation
        """
        return ms, vt1+vt2



class TransferComponent(Component):
    O1_complete = False
    def direct(self, ds, vt_in):
        """
        Direct equation with full time dependence
        """
        return ds, vt_in

    def order1(self, ps, vm_in):
        """
        Equation assuming O(1) periodic input
        """
        return ps, vm_in

    def ordere(self, ms, vt_in):
        """
        Linearized response to O(ε) slow-time input
        """
        return ms, vt_in

    def multiscale(self, ms, vt_in):
        """
        Fully nonlinear multiscale equation in phasor representation
        """
        return ms, vt_in


class SourceComponent(Component):
    O1_complete = False
    def direct(self, ds):
        """
        Direct equation with full time dependence
        """
        pass

    def order1(self, ps):
        """
        Source for O(1) periodic input
        """
        pass

    def ordere(self, ms):
        """
        Source for O(ε) slow-time input
        """
        pass

    def multiscale(self, ms):
        """
        Fully nonlinear multiscale equation in phasor representation
        """
        pass


class LinearComponent(TransferComponent):
    """
    Defines a generic linear component with a transfer function.
    
    Note: a linear component doesn't require the O(1) state to
          calculate the O(ε) solution
    """
    def transfer_function(self, omega):
        raise RuntimeError, "Base class can't be used directly"
        
    def direct(self, ds, vt_in):
        """
        Direct equation with full time dependence
        """
        Hw = self.transfer_function(ds.omega)

        #Force symmetric frequency behaviour
        Vt = ds.fft(vt_in)*Hw
        #Vt[0] = np.real(Vt[0])
        #Vt[-ds.N//2+1:] = np.conj(Vt[1:ds.N//2][::-1])
        vt_out = self.eloss*fftp.ifft(Vt)
        return ds, vt_out

    @timer.time_function(prefix="linear:")
    def multiscale(self, ms, vt_in):
        """
        Fully nonlinear multiscale equation in phasor representation
        """
        Hw = self.transfer_function(ms.omega)
        Vt = ms.fft(vt_in)*Hw
        vt_out = self.eloss*fftp.ifft(Vt)
        return ms, vt_out

    def order1(self, ps, v0_in):
        """
        Equation assuming O(1) periodic input
        """
        Hw = self.transfer_function(ps.omega)
        v0_out = self.eloss*v0_in*Hw

        self.O1_complete = True
        return ps, v0_out

    def ordere(self, ms, v1_in):
        """
        Linearized response to O(ε) slow-time input
        """
        Hw = self.transfer_function(ms.omega)
        v1_out = self.eloss*ms.ifft(ms.fft(v1_in)*Hw)
        return ms, v1_out
        
    def plot_transfer_function(self, ds):
        ax1=pl.subplot(211)
        ax2=pl.subplot(212)
        
        Hw = self.transfer_function(ds.omega)
        ax1.plot(fftp.fftshift(ds.f), np.abs(fftp.fftshift(Hw))**2, 'k--')
        ax2.plot(fftp.fftshift(ds.f), np.angle(fftp.fftshift(Hw)), 'k--')


class Amplifier(LinearComponent):
    """
    Simple amplifier with constant gain
    """
    def __init__(self, g=1.0, iloss=1.0):
        self.set_gain(g)
        self.set_insertion_loss(iloss)

    def set_gain(self, g):
        self.g = g

    def transfer_function(self, omega):
        return self.g


class LorenzianFilter(LinearComponent):
    """
    The simplest single pole filter. This is not a real filter,
    it only has a pole in the positive frequency spectrum, and
    can only act on the phasor representation
    """
    def __init__(self, wf, G, iloss=1.0):
        self.G = pi*G
        self.wf = wf
        self.set_insertion_loss(iloss)

    def transfer_function(self, omega):
        Hwp = -1j*self.G/(omega - self.wf - 1j*self.G)
        return Hwp

class HarmonicResonator(LinearComponent):
    """
    A double pole filter
    Parameters:
    wf - filter central frequency
    Q - Q factor
    """
    def __init__(self, wf, Q, iloss=1.0):
        self.Q = Q
        self.wf = wf
        self.set_insertion_loss(iloss)

    def transfer_function(self, omega):
        Hwp = 1j*omega*self.wf/self.Q/(self.wf**2 - omega**2 + 1j*omega*self.wf/self.Q)
        return Hwp

class LorenzianFilter2(LinearComponent):
    """
    A double pole filter
    Parameters:
    wf - filter central frequency
    G -  the filter bandwith
    """
    def __init__(self, wf, G, iloss=1.0):
        self.G = 2*pi*G
        self.wf = wf
        self.set_insertion_loss(iloss)

    def transfer_function(self, omega):
        Hwp = 1j*omega*self.G/(self.wf**2 - omega**2 + 1j*omega*self.G)
        return Hwp

class FilterZPK(LinearComponent):
    """
    Z - Zeros
    P - Poles
    K - Scale constant
    """
    def __init__(self, Z, P, K, iloss=1.0):
        self.Z = Z
        self.P = P
        self.K = K
        self.set_insertion_loss(iloss)

    def transfer_function(self, omega):
        Hwp = 0
        return Hwp

#signal.butter(2,1.0,btype='low',output='zpk')





class RectangularBandpassFilter(LinearComponent):
    def __init__(self, G, wf, iloss=1.0):
        self.G = 2*pi*G
        self.wf = wf
        self.set_insertion_loss(iloss)

    def transfer_function(self, omega):
        Hwp = np.absolute(omega)<(self.wf+self.G/2)
        Hwm = np.absolute(omega)>(self.wf-self.G/2)
        Hw = np.asarray(Hwp*Hwm, dtype=np.float_)
        return Hw


class RectangularHighpassFilter(LinearComponent):
    def __init__(self, wf, iloss=1.0):
        self.wf = wf
        self.set_insertion_loss(iloss)

    def transfer_function(self, omega):
        Hw = np.asarray(np.absolute(omega)>self.wf, dtype=np.float_)
        return Hw


class RectangularLowpassFilter(LinearComponent):
    def __init__(self, wf, iloss=1.0):
        self.wf = wf
        self.set_insertion_loss(iloss)

    def transfer_function(self, omega):
        Hw = np.absolute(omega)<self.wf
        Hw = np.asarray(Hw, dtype=np.float_)
        return Hw

class Delay(LinearComponent):
    def __init__(self, tao, iloss=1.0):
        self.tao = tao
        self.set_insertion_loss(iloss)

    def transfer_function(self, omega):
        Hw = np.exp(1j*omega*self.tao)
        return Hw
 


