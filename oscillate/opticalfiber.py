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

# ODE Solvers
from scipy import integrate
#import odesolve

import timer

#Speed of light
c = 2.998e8               # - m/s


class GenericFiber(object):
    """
    An optical fiber with user specified:
    beta:   dispersion (a list of b1,b2,b3...)
    gamma:  nonlinear coefficient ()
    length: length of the fiber (m)
    
    Note: All these should be normalized or in SI units
    """
    def __init__(self, alpha=0, beta2=0, gamma=0, length=0):
        self.alpha_value = alpha
        self.beta_value = [0,0,beta2,0]
        self.gamma_value = gamma
        self.length = length

    def set_alpha(self, a):
        self.alpha_value = a

    def set_betas(self, b0=0, b1=0, b2=0, b3=0):
        self.beta_value = [b0,b1,b2,b3]

    def set_gamma(self, gamma):
        self.gamma_value = gamma

    def beta_expansion(self, wl, ns=[2]):
        beta = np.array(self.beta_value)[ns]
        return beta

    def beta(self, wl, omega, ns=[2]):
        b1,b2 = np.array(self.beta_value)[1:3]
        betaw = b1*omega + 0.5*b2*omega**2
        return betaw

    def gamma(self, wl):
        return self.gamma_value

    def alpha(self, wl):
        return self.alpha_value

    def info(self, wl):
        b1,b2,b3 = self.beta_expansion(wl, [1,2,3])
        a_db= self.alpha(wl)
        gamma = self.gamma(wl)
        
        print "Dispersion coefficients:"
        print "β1 = %g ps/km" % (b1/(p_/k_))
        print "β2 = %g ps^2/km" % (b2/(p_**2/k_))
        print "β3 = %g ps^3/km" % (b3/(p_**3/k_))

        print "Loss:"
        print "Loss = %g dB/km" % a_db

        print "Nonlinearity coefficients:"
        print "γ = %g 1/(m-W)" % gamma


    def plot(self,wlrange=[1.2e-6,1.6e-6]):
        wls = np.linspace(wlrange[0],wlrange[1],100)
        omega = 2*pi*3e8/(wls)
        
        betas = np.array([self.beta(wl, [1,2,3]) for wl in wls])
        gammas = np.array([self.gamma(wl) for wl in wls])

        pl.subplot(311)
        pl.plot(wls/u_, betas[:,0], 'b--', label=r'$\beta_1$')
        pl.subplot(312)
        pl.plot(wls/u_, betas[:,1], 'g--', label=r'$\beta_2$')
        pl.subplot(313)
        pl.plot(wls/u_, betas[:,2], 'r--', label=r'$\beta_2$')

        #Integrate:
        D = 0.25*self.S0*(wls-self.lambda0**4/wls**3)
        b1_calc = integrate.cumtrapz(D, wls)
        b1_calc = np.append(b1_calc, b1_calc[-1])

        b_calc = integrate.cumtrapz(b1_calc, omega)
        b_calc = np.append(b_calc, b_calc[-1])
        print b1_calc+betas[0,0]
        
        #betaw = betas[:,0]*omega+betas[:,1]*omega**2+betas[:,2]*omega**3
        pl.subplot(311)
        pl.plot(wls/u_, b1_calc+betas[0,0], 'r-', label=r'$\beta_1 calc$')


class CorningSMF28e(GenericFiber):
    def __init__(self):
        #Zero dispersion wavelength and slope
        self.lambda0 = 1313 * n_        # m
        self.S0 = 0.086 * p_/(n_**2*k_) # s/m^3

        #Interpolated effective index
        self.neff = extrap1d([1310e-9,1550e-9],[1.4677, 1.4682])
        self.dneffdw = (1.4677-1.4682)/(1/1310e-9-1/1550e-9)/(2*pi)

        #Interpolated mode field diameter (m)
        self.mfd = interp1d([1310e-9,1550e-9],[9.2*u_, 10.4*u_], bounds_error=False, kind='linear')

        #interpolated loss (db/m)
        self.loss = interp1d([1310e-9,1550e-9],[0.03/k_, 0.02/k_], bounds_error=False, kind='linear')

        self.pmd = 0.02*p_/np.sqrt(k_)                #ps/√m

        self.n2 = 3e-16*(1e-2**2)
        self.alpha_value = 0
        
    def dispersion_from_wl(self, wl, order=2):
        #Dispersion calculation from the Corning SMF data sheet

        #First order dispersion - s/m
        b1 = (self.neff(wl) + 0*2*pi*c/wl*self.dneffdw)/c

        #Second order dispersion - s^2/m
        b2 = -self.S0*(wl**3 - self.lambda0**4/wl)/(8*pi*c)

        #Third order dispersion - s^3/m
        b3 = self.S0*(3*wl**4 + self.lambda0**4)/(16*pi**2*c**2)

        return np.array([b1,b2,b3])

    def beta(self, wl, n=[2]):
        #Calculate dispersion at wavelength
        beta = self.dispersion_from_wl(wl,3)
        return beta[n]

    def gamma(self, wl):
        mfd = self.mfd(wl)
        Aeff = pi*mfd**2/4
        
        gamma = self.n2*2*pi/(wl*Aeff)
        return gamma




# -----------   Propagation Components  -----------------


class LinearOpticalPropagation(LinearComponent):
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

    def transfer_function(self, omega):
        fiber = self.fibers[0]
        
        alpha = fiber.alpha(self.wl)
        betaw = fiber.beta(self.wl, omega)
        L = fiber.length

        Hw = np.exp(-0.5*alpha*L)
        Hw *= np.exp(1j*betaw*L)
        return Hw


class NonlinearLinearOpticalPropagation(TransferComponent):
    """
    Propagates an optical signal linearly through a fiber
    with periodic boundary conditions.
    
    Arguments:
     * wl:  wavelength
     * fiber_list: a list of fibers to propagate the signal
       through
    """
    def __init__(self, fiber_list, wl=1.0, tol=1e-8):
        self.wl = wl
        self.fibers = fiber_list
        self.set_insertion_loss(1.0)

        self._solve_multiscale = MultiscaleSSF2(tol=tol)
        
        self.tol = tol
        self.store_solution = False
        self.O1_complete = False
        
    def direct(self, ds, vt_in):
        """
        Direct equation with full time dependence
        """
        fiber = self.fibers[0]

        prop = SSF2Adaptive(fiber.alpha(self.wl),
                fiber.beta(self.wl, ds.omega),
                fiber.gamma(self.wl),
                tol=1e-10)
        
        # Optical propagation over fiber length
        vt_out = prop.solve(vt_in, fiber.length)

        return ds, vt_out

    def multiscale(self, ms, v_in):
        """
        Optical multiscale propagation of quasi-periodic input
        
        Here we take the input as a phasor in the optical domain
        and the output is returned as a phasor in the optical domain
        
        vm_out = [a₀, a₁, a₂ ... aᵥ, a₋ᵥ ... a₋₂, a₋₁]

        where the optical signal at optical carrier frequency wc is:
        E(t) = sum_{k=-v}^v a_k exp(j k w0 t) exp(j wc t)

        using the positive frequency convention (as in Rubiola)
        """
        prop = self._solve_multiscale
        fiber = self.fibers[0]

        prop.set_fiber_parameters(
                fiber.alpha(self.wl),
                fiber.beta(self.wl, ms.omega),
                fiber.gamma(self.wl))
        
        # Optical propagation over fiber length
        if self.store_solution:
            zspan = (np.arange(0,101)/100.)*fiber.length
            v_out = prop.solve_and_store(ms, v_in, zspan)
            self.y_store = prop.y_store
            self.z_store = prop.z_store
        else:
            v_out = prop.solve(ms, v_in, fiber.length)

        return ms, v_out


    def order1(self, ps, v0_in):
        """
        Optical propagation of O(1) periodic input
        """
        fiber = self.fibers[0]

        prop = MultiscaleOrder1SSF2(fiber.alpha(self.wl),
                fiber.beta(self.wl, ps.omega),
                fiber.gamma(self.wl),
                tol=self.tol)

        # Optical propagation over fiber length
        v0_out = prop.solve(ps, v0_in, fiber.length)

        return ps, v0_out

    def order1_alt(self, ps, v0_in):
        fiber = self.fibers[0]
        alpha = np.complex128(fiber.alpha(self.wl))
        gamma = np.complex128(fiber.gamma(self.wl))
        betaw = fiber.beta(self.wl, ps.omega).astype(np.complex128)

        #r = odesolve.ode_rk45(df_nls_order1)
        r = integrate.ode(df_nls_order1)
        r.set_integrator('zvode', method='adams', nsteps=5000, atol=1e-12, rtol=1e-12)
        r.set_initial_value(v0_in, 0)
        r.set_f_params(alpha,gamma,betaw)

        #Solve
        Nz = 100
        dz = fiber.length/(Nz-1.0)
        self.O1_z = np.arange(0,Nz)*dz
        self.O1_y = np.zeros(v0_in.shape+(Nz,), np.complex_)

        #Solve to each step in the given range
        self.O1_y[...,0] = v0_in
        for ii in xrange(1,Nz):
            y_out = r.integrate(self.O1_z[ii])
            self.O1_y[...,ii] = y_out

        #Flag O(1) run as done
        self.O1_run_complete = True
        return ps, y_out


    def ordere(self, ms, v1_in):
        """
        Modulator response to O(ε) slow-time input
        linearized about an O(1) behaviour
        """
        # Check for a prior O(1) solution
        assert self.O1_complete, "O(1) solve must be performed before calling O(ε) solve"

        #Fiber parameters
        fiber = self.fibers[0]
        alpha = np.complex128(fiber.alpha(self.wl))
        gamma = np.complex128(fiber.gamma(self.wl))
        betaw = fiber.beta(self.wl, ms.omega).astype(np.complex128)

        #Solver only deals with flat arrays .. how sad
        y_in = ms.fft(v1_in).flatten()

        #Setup solver
        #r = odesolve.ode_rk45(df_nls_ordere, atol=1e-10, rtol=1e-10)
        r = ode(df_nls_ordere)
        r.set_integrator('zvode', method='adams', nsteps=5000, atol=1e-8, rtol=1e-8)
        r.set_initial_value(y_in, 0)
        r.set_f_params(self.O1_y,self.O1_z, alpha,gamma,betaw, v1_in.shape[0],v1_in.shape[1])

        #Solve to the end - no need for intermediate solutions
        v1_out = r.integrate(fiber.length)
        v1_out = ms.ifft(v1_out.reshape(v1_in.shape))
        
        return ms, v1_out



