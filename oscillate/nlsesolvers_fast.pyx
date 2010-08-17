# encoding: utf-8
# cython: profile=True
# filename: nlse_solvers_test.pyx

"""
Library of optical fibers and propagation
-----------------------------------------

Currently there are optical fibers:
GenericFiber(beta, gamma, length)

CorningSMF()

"""
import numpy as np
cimport numpy as np

from scipy import fftpack as fftp

cimport cython

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

cdef inline unsigned int int_wrap(int a, int b, int c):
    return <unsigned int>(a%(c-b) + b)

cdef inline unsigned int index_sum(int a, int b, int c):
    return <unsigned int>(a%(c-b) + b)


def df_linear_interpolate(float z, val_a, z_a):

    #Taken from scipy.interpolate.interp1d

    # 1. Find where in the orignal data, the values to interpolate would be inserted.
    # Note: if z==z_a[m] then searchsorted returns m
    z_new_index = z_a.searchsorted(z)

    # 2. Clip z_new_index so that it between 1 and the length of za
    #    Removes mis-interpolation of z = z_a[0]
    z_new_index = z_new_index.clip(1, len(z_a)-1).astype(int)

    # 3. Calculate the slope of val over the interval of interest
    z1 = z_a[z_new_index]
    z2 = z_a[z_new_index-1]
    val1 = val_a[..., z_new_index]
    val2 = val_a[..., z_new_index-1]

    # 4. Calculate the actual interpolated value.
    slope = (val1-val2) / (z1 - z2)
    val_new = slope*(z-z2) + val2
    return val_new

#
# The NLS equation for a periodic signal with Fourier coefficients y
#
#
@cython.boundscheck(False)
def df_nls_order1(float z,
        np.ndarray[np.complex128_t,ndim=1] y,
        np.complex128_t alpha,
        np.complex128_t gamma,
        np.ndarray[np.complex128_t,ndim=1] betaw):

    cdef:
        Py_ssize_t Ny, N
        Py_ssize_t ii_, jj_, kk_, ll_
        int ii, jj, kk, ll
        np.ndarray[np.complex128_t,ndim=1] ydot
        np.complex128_t bjbkbl, bkbl

    Ny = y.shape[0]
    N = Ny//2
    ydot = np.zeros(Ny, dtype=np.complex128)
    
    for ii in range(-N,N+1):
        ii_ = ii % Ny

        #Nonlinear contributions
        bjbkbl = 0+0j
        for jj in range(-N,N+1):
          jj_ = jj % Ny

          bkbl = 0+0j
          for kk in range(-N+int_max(0,ii-jj), N+1+int_min(0,ii-jj)):
            kk_ = kk % Ny
            ll_ = (-ii+jj+kk) % Ny
            bkbl = bkbl + y[kk_]*y[ll_].conjugate()

          bjbkbl = bjbkbl + y[jj_]*bkbl

        #y' = -αy + iβ(ω)y + iγ|y|²y
        ydot[ii_] = -0.5*alpha*y[ii_] + 1j*betaw[ii_]*y[ii_] + 1j*gamma*bjbkbl

    return ydot

#
# The linearized driven NLS equation for a signal with Fourier coefficients y
#
#
@cython.boundscheck(False)
def df_nls_ordere(float z,
        np.ndarray[np.complex128_t,ndim=1] y_,
        np.ndarray[np.complex128_t,ndim=2] val_a,
        np.ndarray[np.float64_t,ndim=1] z_a,
        np.complex128_t alpha,
        np.complex128_t gamma,
        np.ndarray[np.complex128_t,ndim=2] betaw,
        unsigned int Nms, unsigned int Nt):

    #cdef int Nms = val_a.shape[0]
    cdef unsigned int Nz = val_a.shape[1]
    cdef int N = Nms//2

    cdef np.ndarray[np.complex128_t,ndim=2] y = y_.reshape(Nms,Nt)

    # Check the shapes are correct
    #assert val_a.shape[0]==Nms, "O(1) solution must have the same number of harmonics as O(ε)"
    #assert z_a.shape[0]==Nz, "z vector must be the same length as O(1) solution 2nd axis"

    # Output vector
    cdef np.ndarray[np.complex128_t,ndim=2] ydot = np.zeros((Nms,Nt), dtype=np.complex128)

    cdef int ii, jj, kk, ll
    cdef unsigned int ii_, jj_, kk_, ll_, jjc_, llc_

    # Intermediate stages of NL term calculation
    cdef np.complex128_t aakk, aakkc
    cdef np.ndarray[np.complex128_t,ndim=1] aabjj = np.zeros(Nt, dtype=np.complex128)

    # Interpolate a_m(z)
    cdef np.ndarray[np.complex128_t,ndim=1] az = df_linear_interpolate(z,val_a, z_a)
    
    # The NLSE
    for ii in range(-N,N+1):
        ii_ = ii % Nms

        # Nonlinear contributions
        aabjj.fill(0+0j)
        for jj in range(-N,N+1):
          jj_ = jj % Nms
          jjc_ = (-jj) % Nms

          aakk = 0+0j
          aakkc = 0+0j
          for kk in range(-N+int_max(0,ii-jj), N+1+int_min(0,ii-jj)):
            kk_ = kk % Nms
            ll_ = (ii-jj-kk) % Nms
            llc_ = (-ii+jj+kk) % Nms

            aakk = aakk + az[kk_]*az[llc_].conjugate()
            aakkc = aakkc + az[kk_]*az[ll_]

          # |a+εb|²(a+εb) = |a|²a + ε(2 a a* b + a a b*) + O(ε²)
          aabjj += 2*aakk*y[jj_] + aakkc*y[jjc_].conjugate()

        # y' = -αy + iβ(ω)y + iγ|y|²y
        #ydot[ii_] = -0.5*alpha*y[ii_] + 1j*betaw[ii_]*y[ii_] + 1j*gamma*aabjj
        ydot[ii_] = 1j*gamma*aabjj
    
    return ydot.flatten()
    #return ydot





class nlse_ssf_rk(object):
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
    def __init__(self, alpha, betaw, gamma, Nsteps=1000, tol=1e-8):
        self.alpha = alpha
        self.betaw = betaw
        self.gamma = gamma
        self.tol = tol
        self.Nsteps = Nsteps

        #Debugging flag        
        self._verbose_ = False

    def set_numeric_parameters(self,dz):
        self.dz = dz

    def fnl(self, z, y, g):
        return 1j*g*np.abs(y)**2*y

    def solve(self, ds, y_in, z_span):
        #Start and end z
        if np.iterable(z_span):
            z = np.min(z_span)
            z_max = np.max(z_span)
            max_h = (z_max-z)/np.float(len(z_span))
        else:
            z = 0
            z_max = z_span
            max_h = z_max

        #Fiber parameters
        betaw = self.betaw
        alpha = self.alpha
        gamma = self.gamma
        
        #Step size
        hz = z_max/self.Nsteps
        self.debug_stepsizes = [hz]
        
        ch2 = 0.5*hz; ch3 = 0.5*hz
        ah21 = 0.5*hz; ah32 = 0.5*hz; ah43 = 1.0*hz
        bh1 = hz/6.; bh2 = hz/3.; bh3 = hz/3.; bh4 = hz/6.
        
        #Initial Y
        Y = fftp.fft(y_in)

        #Linear operator
        D_omega = -0.5*alpha + 1j*betaw
        D_s = np.exp(0.5*hz*D_omega)

        kk = 0
        error_max = 0
        while (z+hz/2)<z_max:
            gammak = gamma

            #Linear 1/2 step
            Y *= D_s

            #Nonlinear step by RK4
            y = ds.ifft(Y)
            k1 = self.fnl(z, y, gammak)
            k2 = self.fnl(z + ch2, y + ah21*k1, gammak)
            k3 = self.fnl(z + ch3, y + ah32*k2, gammak)
            k4 = self.fnl(z + hz, y + ah43*k3, gammak)
            y += bh1*k1 + bh2*k2 + bh3*k3 + bh4*k4
            Y = ds.fft(y)

            #Linear 1/2 step
            Y *= D_s
            
            #Update current solution z
            z += hz
            kk += 1
            
        if self._verbose_:
            print "Propagated to %.6g, est err: %.2g, goal err: %.2g, iter: %d"\
             % (z, error_max, self.tol, kk)

        y = fftp.ifft(Y)
        return y


class nlse_multiscale_ssf(nlse_ssf_rk):
    @cython.boundscheck(False)
    def fnl(self, np.float64_t z, np.ndarray[np.complex128_t,ndim=2] y, np.float64_t g):
        cdef int Nms = y.shape[0]
        cdef int Nt = y.shape[1]
        cdef int N = Nms//2
        cdef np.ndarray[np.complex128_t,ndim=2] ydot = np.zeros((Nms,Nt), dtype=np.complex128)

        cdef int ii, jj, kk
        cdef unsigned int ii_, jj_, kk_, ll_

        cdef np.ndarray[np.complex128_t,ndim=1] bkbl = np.zeros(Nt, dtype=np.complex128)
        for ii in range(-N,N+1):
            ii_ = ii % Nms

            #Nonlinear contributions
            for jj in range(-N,N+1):
              jj_ = jj % Nms

              bkbl.fill(0+0j)
              for kk in range(-N+int_max(0,ii-jj), N+1+int_min(0,ii-jj)):
                kk_ = kk % Nms
                ll_ = (-ii+jj+kk) % Nms
                bkbl += y[kk_]*y[ll_].conjugate()

              ydot[ii_] +=  y[jj_]*bkbl
    
        ydot *= 1j*g
        return ydot


