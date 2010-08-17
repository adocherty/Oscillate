
# _*_ coding=utf-8 _*_
#αβγδεζηθικλμνξοπρσςτυφχψω ϕϚϑ ‘’	“”  ·
#ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
#ℋℌℍℎℏℐℑℒℓℕ№℗℘ℙℚℛℜℝℤ℥ℨÅℬℰℱℳℼℽ∇
#¼½¾⅓ ⅔⅕⅖⅗⅘ ⅙⅚ ⅛

"""
Electro-optic modulator classes

Classes
--------

EOMGeneric: 
EOM: 

"""

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi
from scipy import special
from scipy import fftpack as fftp

from components import Component, TransferComponent

# Debugging/profiling
import timer

class EOMGeneric(TransferComponent):
    """
    Models the optical output for a single electric input
    of a MZM using the following general expression:
    
    E(t) = Em [ exp(j v1 A(t)) + a exp(j v2 A(t) + j phi) ]
    
    The input is the modulating voltage, the output the
    electric field strength of the modulated light

    eom = EOM_generic(v1, v2, a, phi)
    
    Parameters
    ----------
    Em : float
        Amplitude
    v1,v2 : float
        Branch amplitudes
    a : float
        Mixing ratio
    phi : float
        Branch optical phase difference    
    """
    def __init__(self, Em, v1, v2, a, phi, iloss=1.0):
        self.v1,self.v2 = v1,v2
        self.a,self.phi = a,phi
        self.Em = Em
        self.set_insertion_loss(iloss)

    def direct(self, ds, vt_in):
        """
        Direct equation with full time dependence
        """
        t, omega = ds.t, ds.omega
        
        # Nonlinear modulator response function
        b1 = np.exp(1j*self.v1*vt_in)
        b2 = np.exp(1j*self.v2*vt_in+1j*self.phi)
        et_mod = self.eloss*self.Em*(b1 + self.a*b2)

        return ds, et_mod

    @timer.time_function(prefix="eom:")
    def multiscale(self, ms, vm_in):
        """
        Here we take the input as an phasor in the electrical RF domain
        and the output is returned as a phasor in the optical domain
        
        vm_in = [a0, a1, a2 ... an]
        where
        v(t) = sum_{k=0}^n a_k exp(j k w0) + c.c.

        vm_out = [a0, a1, a2 ... an, a-n ... a-1]
        where
        E(t) = sum_{k=-n}^n a_k exp(j k w0)

        Note:
        Currently only the dc and fundamental component of the signal are used.
        All other harmonics are assumed filtered out.
        """
        #Output signal will be complex
        mso = ms.copy()
        mso.real = False
        
        #Assume only fundamental harmonic and DC at input
        a0 = vm_in[0]
        a1,theta1 = np.abs(vm_in[1]), np.angle(vm_in[1])

        #Jacobi-Anger expansion
        k = mso.ms[:,np.newaxis]
        b1 = special.jv(k, self.v1*a1) * np.exp(1j*self.v1*a0 + 1j*k*(theta1+0.5*pi))
        b2 = special.jv(k, self.v2*a1) * np.exp(1j*self.phi + 1j*self.v2*a0 + 1j*k*(theta1+0.5*pi))
        
        vm_out = self.eloss*self.Em*(b1 + self.a*b2)

        return mso, vm_out

    @timer.time_function(prefix="eom:")
    def order1(self, ps, v0_in):
        """
        Equation assuming O(1) periodic input
        
        Here we take the input as an phasor in the electrical RF domain
        and the output is returned as a phasor in the optical domain
        
        vm_in = [a0, a1, a2 ... an]
        where
        v(t) = sum_{k=0}^n a_k exp(j k w0) + c.c.

        vm_out = [a0, a1, a2 ... an, a-n ... a-1]
        where
        E(t) = sum_{k=-n}^n a_k exp(j k w0)

        Note:
        Currently only the dc and fundamental component of the signal are used.
        All other harmonics are assumed filtered out.
        """
        #Output signal will be complex
        pso = ps.copy()
        pso.real = False
        
        #Assume only fundamental harmonic and DC at input
        a0 = v0_in[0]
        a1,theta1 = np.abs(v0_in[1]), np.angle(v0_in[1])

        #Jacobi-Anger expansion
        k = pso.ms
        b1 = special.jv(k, self.v1*a1) * np.exp(1j*self.v1*a0 + 1j*k*(theta1+0.5*pi))
        b2 = special.jv(k, self.v2*a1) * np.exp(1j*self.phi + 1j*self.v2*a0 + 1j*k*(theta1+0.5*pi))
        bp1 = special.jvp(k, self.v1*a1) * np.exp(1j*self.v1*a0 + 1j*k*(theta1+0.5*pi))
        bp2 = special.jvp(k, self.v2*a1) * np.exp(1j*self.phi + 1j*self.v2*a0 + 1j*k*(theta1+0.5*pi))
        
        v0_out = self.eloss*self.Em*(b1 + self.a*b2)

        #Flag and store the O1 solution
        self.O1_solution = (ps,v0_in,b1,b2,bp1,bp2)
        self.O1_complete = True

        return pso, v0_out

    @timer.time_function(prefix="eom:")
    def ordere(self, ms, v1_in):
        """
        Modulator response to O(ε) slow-time input
        linearized about an O(1) behaviour
        """
        assert ms.real, "Modulator input signal must be real"

        assert self.O1_complete, "O(1) simulation must be complete before O(ε) can be run"
        (ps,v0_in,b1,b2,bp1,bp2) = self.O1_solution
        
        #Output signal will be complex
        mso = ms.copy()
        mso.real = False
        
        #O(1) bias point
        a0 = v0_in[0]
        a1,theta1 = np.abs(v0_in[1]), np.angle(v0_in[1])

        #O(ε) signal
        be0 = v1_in[0]
        be1, eta1 = np.abs(v1_in[1]), np.angle(v1_in[1])
        c1 = be1*np.cos(eta1-theta1)
        d1 = be1/a1*np.sin(eta1-theta1)

        # Linearized modulator expansion about O(1) operation point
        k = mso.ms[:,np.newaxis]
        v1_out = (self.v1*bp1 + self.a*self.v2*bp2)[:,np.newaxis]*c1
        v1_out += 1j*k*(b1[:,np.newaxis] + self.a*b2[:,np.newaxis])*d1
        v1_out += (b1 + self.a*b2)[:,np.newaxis]*be0
        
        return mso, self.eloss*self.Em*v1_out


class EOM(EOMGeneric):
    """
    The input is the modulating voltage, the output the
    electric field strength of the modulated light

    Parameters
    ----------
    P0 : float 
        input optical power (W)
    Vb : float
        Bias voltage
    Vpi : float
        1/2 Wave Voltage
    alpha : float
        chirp paramter
    eta : float
        modulation efficiency
    iloss : float
        fractional insertion loss
    """
    def __init__(self, P0=1.0, eta=1.0, alpha=0.0, Vb=3.14, Vpi=3.14, iloss=1.0):
        v1 = 0.5*pi*(alpha-1)/Vpi
        v2 = 0.5*pi*(alpha+1)/Vpi
        phi = 0.5*pi + pi*Vb/Vpi

        #Modulation strength and amplitude
        a = (1-np.sqrt(1-eta**2))/eta
        Em = np.sqrt(0.5*P0/(1+a**2))

        EOMGeneric.__init__(self, Em, v1, v2, a, phi, iloss)


class EOMPhysical(EOMGeneric):
    """
    EOM with parameters in physical units
    
    Parameters
    ----------
    P0 : float 
        input optical power (W)
    er : flaot
        extinction ratio (ratio)
    alpha : float
        chirp paramter
    Vb : float
        Bias voltage
    Vpi : float
        1/2 Wave Voltage
    iloss : float
        fractional insertion loss
    """
    def __init__(self, P0=1.0, er=1.0, alpha=0.0, Vb=3.14, Vpi=3.14, iloss=1.0):
        v1 = 0.5*pi*(alpha-1)/Vpi
        v2 = 0.5*pi*(alpha+1)/Vpi
        phi = 0.5*pi + pi*Vb/Vpi

        #Modulation strength
        eta = (er-1)/(er+1)
        a = (1-np.sqrt(1-eta**2))/eta
    
        #Optical input amplitude
        Em = np.sqrt(0.5*P0/(1+a**2))
        EOMGeneric.__init__(self, Em, v1, v2, a, phi, iloss)


class EOMCovegaMach10(EOMPhysical):
    """
    Mach-10™ 057/063
    -0.7 Fixed-Chirp Intensity Modulator with integrated Photodetector

    This module models the modulator alone
    
    Parameters
    -----------
    P0 : float
        input optical power (W)
    Vb : float
        Bias voltage (V)
    Vpi : float
        1/2 Wave Voltage (V)
    """
    def __init__(self, P0=1.0, Vb=3.14, Vpi=3.14):
        # Optical extinction ratio at DC: >20 dB
        # Optical extinction ration PRBS: >13 dB
        er = db_to_signal(13)
        
        # Insertion loss: 4 - 5 dB
        iloss = db_to_power(-4)
        
        # Modulator chirp parameter: 0.5 - 0.8 
        alpha = 0.7
        
        EOMPhysical.__init__(self, P0,er,alpha,Vb,Vpi,iloss)


class ModulatorDetector(TransferComponent):
    """
    The model of a MZM and optical power detector with no
    effects of optical propagation.
    
    This allows operating the model with fewer harmonics than
    are needed for full optical simulations.
    
    v_{out}(t) = Vph {1 - η sin [π (Vin(t) + Vb)/Vpi] }
    
    where Vph = P0 
    
    The input is the modulating voltage, the output the
    detected voltage.

    Parameters
    ----------
    Vb : float
        Bias voltage
    Vpi : float
        1/2 Wave Voltage
    Vph : float
        The photodetector voltage Vph = P0 rho R/2
    eta : float
        The fractional response
    iloss : float
        fractional insertion loss
    """
    def __init__(self, eta=1.0, Vb=3.14, Vpi=3.14, Vph=40., iloss=1.0):
        self.Vpi = Vpi
        self.Vb = Vb
        self.Vph = Vph
        self.eta = eta
        
        self.set_insertion_loss(iloss)

    def direct(self, ds, u_in):
        """
        Direct equation with full time dependence
        """
        u_out = self.eloss*self.Vph*(1. - self.eta*np.sin(pi*(u_in) + self.Vb)/self.Vpi)
        return ds, u_out

    @timer.time_function(prefix="modet:")
    def multiscale(self, ms, vm_in):
        """
        Here we take the input as an phasor in the electrical RF domain
        and the output is returned as a phasor in the optical domain
        
        vm_in = [a0, a1, a2 ... an]
        where
        v(t) = sum_{k=0}^n a_k exp(j k w0) + c.c.

        vm_out = [a0, a1, a2 ... an, a-n ... a-1]
        where
        E(t) = sum_{k=-n}^n a_k exp(j k w0)

        Note:
        Currently only the dc and fundamental component of the signal are used.
        All other harmonics are assumed filtered out.
        """
        #Assume only fundamental harmonic and DC at input
        a0 = vm_in[0]
        a1,theta1 = np.abs(vm_in[1]), np.angle(vm_in[1])

        #Jacobi-Anger expansion
        #
        # v_out = Vph {1 + i η/2 J_0(z1) cos(z0) + i sum_{k=1}^inf i^k J_k(z1) 
        #              * [exp(i z0)-(-1)^k exp(i z0)] exp(i k w0 t) }
        #
        # z0 = π (a0+Vb)/Vπ
        # z1 = π a1/Vπ
        k = ms.ms[:,np.newaxis]
        z0 = pi/self.Vpi*(a0+self.Vb)
        z1 = pi*a1/self.Vpi

        b1 = np.exp(1j*z0 + 1j*k*(theta1+0.5*pi))
        b2 = np.exp(-1j*z0 + 1j*k*(theta1-0.5*pi))
        
        vm_out = 1j*self.eta*special.jv(k, z1)*(b1 - b2)
        vm_out[0] *= 0.5
        vm_out[0] += 1.0
        return ms, self.eloss*self.Vph*vm_out

    @timer.time_function(prefix="modet:")
    def order1(self, ps, v0_in):
        """
        Equation assuming O(1) periodic input
        
        Here we take the input as an phasor in the electrical RF domain
        and the output is returned as a phasor in the optical domain
        
        vm_in = [a0, a1, a2 ... an]
        where
        v(t) = sum_{k=0}^n a_k exp(j k w0) + c.c.

        vm_out = [a0, a1, a2 ... an, a-n ... a-1]
        where
        E(t) = sum_{k=-n}^n a_k exp(j k w0)

        Note:
        Currently only the dc and fundamental component of the signal are used.
        All other harmonics are assumed filtered out.
        """
        #Assume only fundamental harmonic and DC at input
        a0 = v0_in[0]
        a1,theta1 = np.abs(v0_in[1]), np.angle(v0_in[1])

        #Jacobi-Anger expansion
        #
        # v_out = Vph {1 + i η/2 J_0(z1) cos(z0) + i sum_{k=1}^inf i^k J_k(z1) 
        #              * [exp(i z0)-(-1)^k exp(i z0)] exp(i k w0 t) }
        #
        # z0 = π (a0+Vb)/Vπ
        # z1 = π a1/Vπ
        k = ps.ms
        z0 = pi/self.Vpi*(a0+self.Vb)
        z1 = pi*a1/self.Vpi
        
        b1 = np.exp(1j*z0 + 1j*k*(theta1+0.5*pi))
        b2 = np.exp(-1j*z0 + 1j*k*(theta1-0.5*pi))
        
        v0_out = 1j*self.eta*special.jv(k, z1)*(b1 - b2)
        v0_out[0] *= 0.5
        v0_out[0] += 1.0
        return ps, self.eloss*self.Vph*v0_out

    @timer.time_function(prefix="modet:")
    def ordere(self, ms, v1_in):
        """
        Modulator response to O(ε) slow-time input
        linearized about an O(1) behaviour
        """

        return None


