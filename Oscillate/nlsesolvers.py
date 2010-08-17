# encoding: utf-8
#αβγδεζηθικλμνξοπρσςτυφχψω ϕϚϑ ‘’	“”  ·
#ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
#ℋℌℍℎℏℐℑℒℓℕ№℗℘ℙℚℛℜℝℤ℥ℨÅℬℰℱℳℼℽ∇√
#¼½¾⅓ ⅔⅕⅖⅗⅘ ⅙⅚ ⅛

"""
Library of optical fibers and propagation
-----------------------------------------

Currently there are optical fibers:
GenericFiber(beta, gamma, length)

CorningSMF()

"""
import time, sys

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi
from numpy import linalg as la
from scipy import random, signal, integrate
from scipy import fftpack as fftp
from scipy.interpolate import interp1d

try:
    import specializedconvolve as sc
except:
    print "Warning: optical propagation not available without compilation"
    #import slowconvolve as sc

class SSF2Adaptive(object):
    """
    Propagates an optical signal through a fiber
    with periodic boundary conditions using
    the split step Fourier transform
    
    Arguments:
     * alpha: power loss coefficient
     * betaw: dispersion relation, over frequency
              typically: - D**2
     * gamma: nonlinear .. constant
     
    """ 
    def __init__(self, alpha, betaw, gamma, tol=1e-8):
        self.alpha = alpha
        self.betaw = betaw
        self.gamma = gamma
        self.tol = tol
        self.current_z = 0

        #Debugging flag
        self._verbose_ = 1
        self._debug_ = False

    def set_numeric_parameters(self,dz):
        self.dz = dz

    def _linear_step(self, Y, Dh):
        Y *= Dh
        return Y

    def _nonlinear_step(self, Y, gh):
        y = fftp.ifft(Y)
        y *= np.exp(1j*gh*np.abs(y)**2)
        Y = fftp.fft(y)
        return Y

    def solve_and_store(self, y_in, z_span):
        # Default solve if z_span is not iterable
        if not np.iterable(z_span):
            return self.solve(y_in, z_span)

        z = np.min(z_span)
        z_max = np.max(z_span)
        max_h = (z_max-z)/np.float(len(z_span))
        
        self.y_store = np.zeros(z_span.shape + y_in.shape, np.complex_)

        y = y_in
        for ii in range(len(z_span)):
            y = self.solve(y,z_span[ii],z)
            self.y_store[ii] = y

            z = z_span[ii]
            iistore += 1

        return y

    def solve(self, y_in, z_max, z_0=None):
        #Fiber parameters
        betaw = self.betaw
        alpha = self.alpha
        gamma = self.gamma
        
        #initial step size selection
        #this should be automatic
        z = self.current_z if z_0 is None else z_0
        max_h = (z_max-z)/10.0
        hz = (z_max-z)/100.0

        if self._debug_: self.debug_stepsizes = [hz]
        
        #Initial Y
        Y = fftp.fft(y_in)
        D_omega = -0.5*alpha + 1j*betaw

        kk = 0
        iistore = 1
        hz_changed = True
        error_max = 0
        while (z+hz/2)<z_max:
            kk += 1
            gammak = gamma #*np.exp(-alpha*z)
            
            #Update Linear operators
            if hz_changed:
                D_1s = np.exp(0.25*hz*D_omega)
                D_2s = np.exp(0.5*hz*D_omega)

            #Single step of 2h
            Y2 = Y.copy()
            Y2 = self._linear_step(Y2, D_2s)
            Y2 = self._nonlinear_step(Y2, hz*gammak)
            Y2 = self._linear_step(Y2, D_2s)

            #Two steps of h
            Y1 = Y.copy()
            Y1 = self._linear_step(Y1, D_1s)
            Y1 = self._nonlinear_step(Y1, 0.5*hz*gammak)
            Y1 = self._linear_step(Y1, D_2s)
            Y1 = self._nonlinear_step(Y1, 0.5*hz*gammak)
            Y1 = self._linear_step(Y1, D_1s)
            
            #Update current solution z
            z_next = z + hz

            #Error estimate
            error = la.norm(Y2-Y1)/la.norm(Y1)

            #Adaptively update step size
            hz_changed = True
            accept_current_z = True
            if error>2*self.tol:
                hz /= 2
                accept_current_z = False
            elif error>self.tol:
                hz /= 2**(1./3.)
            elif error<self.tol/2:
                hz *= 2**(1./3.)
            else:
                hz_changed = False

            #Limit the size of h
            #Currently, just to make sure storing the
            #solution works
            if hz_changed and hz>max_h:
                hz = max_h

            #Construct O(h^4) accurate solution
            if accept_current_z:
                Y = (4./3.)*Y1-(1./3.)*Y2
                z = z_next
                error_max = max(error, error_max)

            if self._debug_: self.debug_stepsizes += [hz]

            #Final step shouldn't take us over z_max
            if (z_max-z)<hz:
                hz = (z_max-z)
                hz_changed = True
            
        if self._verbose_:
            print "Propagated to %.6g, est err: %.2g, goal err: %.2g, iter: %d"\
             % (z, error_max, self.tol, kk)

        y = fftp.ifft(Y)
        return y



class SSF2RK2(object):
    """
    Propagates an optical signal through a fiber
    with periodic boundary conditions using
    the 2nd order split-step Fourier method
    
    Arguments:
     * alpha: power loss coefficient
     * betaw: dispersion relation, over frequency
              typically: - D**2
     * gamma: nonlinear .. constant
     
    """ 
    def __init__(self, alpha=None, betaw=None, gamma=None, tol=1e-8):
        self.tol = tol
        self.current_z = 0

        self.set_fiber_parameters(alpha, betaw, gamma)
        self.set_numeric_parameters()

        #Debugging flag        
        self._verbose_ = 1


    def set_fiber_parameters(self, alpha=None, betaw=None, gamma=None):
        self.alpha = alpha
        self.betaw = betaw
        self.gamma = gamma

    def set_numeric_parameters(self, stepsize=0.05, adaptivestep=1):
        self.stepsize = stepsize
        self.maxstepsize = 1000.
        self.maxsteps = 1e5
        self.adaptivestep = adaptivestep

    def _nonlinear_step(self, z, y, gammak, h):
        # Explicit trapezoidal method or a RK2
        k1 = self.fnl(z, y, gammak)
        k2 = self.fnl(z + h, y + h*k1, gammak)
        y += 0.5*h*(k1 + k2)
        return y
        
    def fnl(self, z, y, g):
        return 1j*g*np.abs(y)**2*y

    def solve_and_store(self, ds, y_in, z_span):
        # Default solve if z_span is not iterable
        if not np.iterable(z_span):
            return self.solve(ds, y_in, z_span)

        z = np.min(z_span)
        z_max = np.max(z_span)
        max_h = (z_max-z)/np.float(len(z_span))
        
        self.y_store = np.zeros(z_span.shape + y_in.shape, np.complex_)
        self.z_store = np.zeros(z_span.shape)

        y = y_in
        self.y_store[0] = y
        self.z_store[0] = z
        for ii in range(1, len(z_span)):
            z_old = z
            z = z_span[ii]
            if z_span[ii]>0:
                y = self._solve(ds,y,z,z_old)

            self.y_store[ii] = y
            self.z_store[ii] = z

        if self._verbose_>0:
            print "\r\x1b[KPropagated to %.6g, est err: %.2g, steps: %g" \
                    % (z_max, self._ir_error_max, self._ir_nsteps)

        return y

    def solve(self, ds, y_in, z_max, z_0=None):
        y = self._solve(ds, y_in, z_max, z_0=z_0)

        if self._verbose_>0:
            print "\r\x1b[KPropagated to %.6g, est err: %.2g, steps: %g" \
                    % (z_max, self._ir_error_max, self._ir_nsteps)
        return y

    def _solve(self, ds, y_in, z_max, z_0=None):
        #Fiber parameters
        betaw = self.betaw
        alpha = self.alpha
        gamma = self.gamma
        
        #Initial step size
        z_start = self.current_z if z_0 is None else z_0
        hz = self.stepsize
        max_h = self.maxstepsize

        #Loop control flags
        hz_changed = True
        final_step = False

        #Final step shouldn't take us over z_max
        if (z_max-z_start)<hz:
            hz = (z_max-z_start)
            final_step = True

        #Initial Y
        Y = ds.fft(y_in)

        #Linear operator
        D_omega = -0.5*alpha + 1j*betaw
        
        #Debug output
        fd=sys.stderr

        #Loop
        z = z_start
        kk = 0
        maxsteps = 1e5
        error_max = 0
        update_adaptive_step = False
        while (z+hz/2)<z_max and kk<=maxsteps:
            gammak = gamma

            #Update Linear operators
            if hz_changed:
                D_s = np.exp(0.5*hz*D_omega)
                D_2s = np.exp(0.25*hz*D_omega)
                hz_changed = False
            
            #Do adaptive steps for the first 10 steps to correctly select step size
            #and then every adaptivestep afterwards
            if self.adaptivestep is not None:
                update_adaptive_step = (np.mod(kk,self.adaptivestep)==0) or (kk<10)

            #Solution with two steps of h/2
            if update_adaptive_step:
                Y_old = Y.copy()
                Y2 = Y.copy()
                Y2 *= D_2s
                y2 = ds.ifft(Y2)
                y2 = self._nonlinear_step(z, y2, gammak, 0.5*hz)
                Y2 = ds.fft(y2)
                Y2 *= D_s
                y2 = ds.ifft(Y2)
                y2 = self._nonlinear_step(z, y2, gammak, 0.5*hz)
                Y2 = ds.fft(y2)
                Y2 *= D_2s

            Y *= D_s            #Linear step

            #Nonlinear step in time domain
            y = ds.ifft(Y)
            y = self._nonlinear_step(z, y, gammak, hz)
            Y = ds.fft(y)

            Y *= D_s            #Linear step

            #Update current solution z
            z_next = z + hz

            #Error test            
            accept_current_z = True
            if update_adaptive_step:
                #Error estimate
                error = la.norm(Y2-Y)/la.norm(Y)

                #Adaptively update step size
                hz_changed = True
                if error>2*self.tol:
                    hz /= 2
                    accept_current_z = False
                elif error>self.tol:
                    hz /= 2**(1./3.)
                elif error<self.tol/2:
                    hz *= 2**(1./3.)
                else:
                    hz_changed = False

                if hz_changed and hz>max_h:
                    hz = max_h
                
                #Store changed stepsize
                self.stepsize = hz

                #Accept solution
                if accept_current_z:
                    Y = (4./3.)*Y2-(1./3.)*Y
                    error_max = max(error, error_max)
                else:
                    Y = Y_old

                if self._verbose_>1:
                    fd.write("\r @ z=%.4f, N=%d, error=%.2e, dz=%.3g  " \
                             % (z, kk, error, hz))

            #Update current z
            if accept_current_z:
                z = z_next

            #Final step shouldn't take us over z_max
            if (z_max-z)<hz:
                hz = (z_max-z)
                hz_changed = True
                #final_step = True
            
            kk += 1
            if kk>maxsteps:
                raise RuntimeError, "SSF exceeded maximum number of steps"
            
        #Store error
        self._ir_nsteps = kk
        self._ir_error_max = error_max
        y = ds.ifft(Y)
        return y

class SSF2RK4(SSF2RK2):
    def _nonlinear_step(self, z, y, gammak, hz):
        ch2 = 0.5*hz; ch3 = 0.5*hz
        ah21 = 0.5*hz; ah32 = 0.5*hz; ah43 = 1.0*hz
        bh1 = hz/6.; bh2 = hz/3.; bh3 = hz/3.; bh4 = hz/6.

        k1 = self.fnl(z, y, gammak)
        k2 = self.fnl(z + ch2, y + ah21*k1, gammak)
        k3 = self.fnl(z + ch3, y + ah32*k2, gammak)
        k4 = self.fnl(z + hz, y + ah43*k3, gammak)
        y += bh1*k1 + bh2*k2 + bh3*k3 + bh4*k4
        return y

class MultiscaleSSF2(SSF2RK2):
    def fnl(self, z, y, g):
        ydot = sc.df_triple_convolve_2d(y,y.conjugate())
        ydot *= 1j*g
        return ydot

class MultiscaleSSF24(SSF2RK4):
    def fnl(self, z, y, g):
        ydot = sc.df_triple_convolve_2d(y,y.conjugate())
        ydot *= 1j*g
        return ydot

#from _generated_tripleproduct import *
#from _generated_tripleproduct_exp import *
class MultiscaleSSF2gen(SSF2RK2):
    def fnl(self, z, y, g):
        #ydot = direct_triple_product_aab4(y,y.conjugate())
        ydot = direct_triple_product_abc4(y,y,y.conjugate())
        ydot *= 1j*g
        return ydot

class MultiscaleOrder1SSF2(SSF2RK2):
    def fnl(self, z, y, g):
        ydot = sc.df_triple_convolve_1d(y,y.conjugate())
        ydot *= 1j*g
        return ydot

class SSF2Exact(SSF2RK2):
    def _nonlinear_step(self, z, y, g, h):
        y *= np.exp(1j*g*h*np.abs(y)**2)
        return y

