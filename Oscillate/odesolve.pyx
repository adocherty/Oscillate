# encoding: utf-8
# cython: profile=False
# filename: odesolve.pyx

# Implementation of ODE Solvers in Cython

import numpy as np
cimport numpy as np
cimport cython

import time

from numpy import linalg as la

#@cython.profile(False)
class ode_rk4:
    def __init__(self, function, T0, y0, Nsteps=1000):
        self.f = function

        self.T = float(T0)
        self.y = np.copy(y0)
        self.Nstep = Nsteps
        
        self.Ny = np.size(self.y)
        self.dtype = self.y.dtype

    def set_f_params(self, *args):
        self.fparams = args

    def integrate(self, Tend):
        #Starting step size
        h = (Tend-self.T)/self.Nsteps
        args = self.fparams
        
        #Inter-step locations and weights
        ch2 = 0.5*h; ch3 = 0.5*h
        ah21 = 0.5*h; ah32 = 0.5*h; ah43 = 1.0*h
        bh1 = h/6.; bh2 = h/3.; bh3 = h/3.; bh4 = h/6.

        Ti = self.T
        yi = self.y

        k4 = self.f(Ti, yi)
        nstep = 0
        while Ti+h/2<Tend:
            k1 = k4
            k2 = self.f(Ti + ch2, yi + ah21*k1, *args)
            k3 = self.f(Ti + ch3, yi + ah32*k2, *args)
            k4 = self.f(Ti + h, yi + ah43*k3, *args)
            
            yi += bh1*k1 + bh2*k2 + bh3*k3 + bh4*k4
            Ti += h
            
            nstep += 1

        return yi


class ode_rk45:
    """
    Implementation of the Dormand-Prince Runge-Kutta 4(5) algorithm

    """

    def __init__(self, function, h_max=None, h_min=None, atol=1e-6, rtol=1e-6):
        self.f = function
        self.atol, self.rtol = atol, rtol
        self.h_max = h_max
        self.h_min = h_min
        
        self._verbose_ = False

    def set_initial_value(self, y0, T0=0):
        self.T = float(T0)
        self.y = np.copy(y0)

        self.Ny = np.size(self.y)
        self.dtype = self.y.dtype

    def set_f_params(self, *args):
        self.fparams = args
        
    def integrate(self, Tend):
        fargs = self.fparams
        Tend = float(Tend)
        h_min,h_max = self.h_min,self.h_max
        
        #Guess starting step size
        if h_max is not None:
            h = h_max/10.0
        else:
            h = (Tend-self.T)/1000.0

        #Inter-step locations
        c2 = 0.2; c3 = 0.3; c4 = 0.8; c5 = 8./9; c6 = 1.0

        #Dormand-Price weights
        a21=1./5
        a31=3./40; a32=9./40
        a41=44./45; a42=-56./15; a43=32./9
        a51=19372./6561; a52=-25360./2187; a53=64448./6561; a54=-212./729
        a61=9017./3168; a62=-355./33; a63=46732./5247; a64=49./176; a65=-5103./18656
        a71=35./384; a72=0.; a73=500./1113; a74=125./192; a75=-2187./6784; a76=11./84

        #Coefficients for RK final step
        b1=5179./57600; b2=0.; b3=7571./16695; b4=393./640; b5=-92097./339200; b6=187./2100; b7=1./40
        bb1=35./384; bb2=0.; bb3=500./1113; bb4=125./192; bb5=-2187./6784; bb6=11./84; bb7=0.
        
        #Coefficients for the local error calculation
        be1=bb1-b1; be2=bb2-b2; be3=bb3-b3; be4=bb4-b4; be5=bb5-b5; be6=bb6-b6; be7=bb7-b7

        #Adaptive step size setup (from Dormand & Prince)
        err_beta = 0.0
        err_exp = 0.2-err_beta*0.75
        hrmin,hrmax = 0.1,5.0     #Range of acceptable stepsizes
        
        Tstore = Ti = self.T
        yi = self.y
        k1 = self.f(Ti, yi,*fargs)


        time_ode = time.time()
        nsteps = 0
        nstore = 0
        hchanged = True
        while Ti+h/2<Tend:

            k2 = self.f(Ti + c2*h, yi + h*a21*k1,*fargs)
            k3 = self.f(Ti + c3*h, yi + h*(a31*k1 + a32*k2),*fargs)
            k4 = self.f(Ti + c4*h, yi + h*(a41*k1 + a42*k2 + a43*k3),*fargs)
            k5 = self.f(Ti + c5*h, yi + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4),*fargs)
            k6 = self.f(Ti + h, yi + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5),*fargs)
            k7 = self.f(Ti + h, yi + h*(a71*k1 + a72*k2 + a73*k3 + a74*k4 + a75*k5 + a76*k6),*fargs)

            #New estimate (with local interpolation) 
            yi_new = yi + h*(b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6 + b7*k7)

            #Error estimate
            ei = np.abs(h*(be1*k1 + be2*k2 + be3*k3 + be4*k4 + be5*k5 + be6*k6 + be7*k7))
            k1 = k7

            #Adaptive step size
            stol = self.atol + self.rtol*np.fmax(np.abs(yi), np.abs(yi_new))
            relerr = np.sqrt(np.sum((ei/stol)**2)/self.Ny)
            err_fac = relerr**err_exp
            hrest = 0.9/err_fac #lerr_old**err_beta/err_fac
            
            #New step size
            hnew = h*min(hrmax,max(hrmin,hrest))

            #print "Est error: ", relerr
            #print "h size: ", hnew

            #Error checking
            if np.any(np.isnan(yi_new)) or np.any(np.isinf(yi_new)):
                raise RuntimeError, "NaN value encountered in ODE solve"

            #Accept solution
            nsteps+=1
            if (relerr<1.0):
                Ti += h
                yi = yi_new

            #Bound step sizes
            hnew = min(hnew, h_max) if h_max else hnew
            hnew = max(hnew, h_min) if h_min else hnew
            
            #Ensure the step size doesn't go beyond the end
            h = min(hnew, Tend-Ti)

            ctos = time.time() - time_ode
            etos = ctos*Tend/Ti
            if self._verbose_ and np.mod(nsteps,1000)==0:
                print "Steps: %d,  Average step size: %.2g,  Time:  %2f/%2f s" \
                    % (nsteps, (Ti/nsteps), ctos, etos)

        if self._verbose_ and nsteps>0:
            print "Total steps:", nsteps, "Average step size:", ((Tend-self.T)/nsteps)

        self.T = Ti
        self.y = yi
        return yi


