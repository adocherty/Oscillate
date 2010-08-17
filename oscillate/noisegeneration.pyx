# encoding: utf-8
# cython: profile=False
# filename: odesolve.pyx

# Implementation of ODE Solvers in Cython

import numpy as np
cimport numpy as np

import scipy as sp

cimport cython

from numpy import pi

def discrete_white_noise(N,dt,Q):
    #Discrete noise variance
    Qd = Q/dt

    # Generate some white noise with σ=1
    w = sp.random.normal(0, np.sqrt(Qd), N)
    return w


def discrete_markov_process(N,dt,Q,a):
    #Discrete noise variance
    Qd = Q*(1-np.exp(-2*dt/a))/(2*a)
    
    #Generate some white noise with σ=Qd
    w = sp.random.normal(0, np.sqrt(Qd), N)

    #Compute the Markov SDE
    # x_{k+1} = x_k + w_k
    x = w
    for kk in range(N-1):
        x[kk+1] = np.exp(-dt/a)*x[kk] + x[kk+1]

    return x

def discrete_brownian_process(N,dt,Q):
    #Discrete noise variance
    Qd = Q*dt
    
    #Generate some white noise with σ=Qd
    w = np.sqrt(Qd)*sp.random.standard_normal(N)

    #Compute the Markov SDE
    # x_{k+1} = x_k + w_k
    x = np.zeros(N)
    x[0] = w[0]
    for kk in range(N-1):
        x[kk+1] = x[kk] + w[kk+1]

    return x

def discrete_falpha_process(N, dt, Q, alpha):
    #Discrete noise variance
    Qd = Q*dt**(alpha-1.0)

    #Generate some white noise with σ=Qd
    w = np.sqrt(Qd)*sp.random.standard_normal(N)

    #Generate the coefficients of the 1/α process
    h = np.zeros(N)
    h[0] = 1.0
    for kk in range(1,N):
        h[kk] = (0.5*alpha+kk-1)*h[kk-1]/kk
    
    #Compute the convolution x_k = h_k ⊛ w_k
    x = np.convolve(h, w, 'full')[:N]
    return x

