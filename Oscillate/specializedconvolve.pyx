# encoding: utf-8
# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# filename: nlse_solvers_test.pyx

import numpy as np
cimport numpy as np

cimport cython

# 
# The python numeric type and c numeric type
# Options: np.complex128 np.float64
#

data_dtype = np.complex128
ctypedef np.complex128_t data_t

#
# Fast functions for indexex
#
#

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

cdef inline Py_ssize_t int_wrap(int a, unsigned int b):
    if a>=0:
        return <Py_ssize_t>(a)
    else:
        return <Py_ssize_t>(a+b)

#
# The convolution for a periodic signal with Fourier coefficients a, b
#
#
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef df_convolve(
    np.ndarray[data_t,ndim=2] a,
    np.ndarray[data_t,ndim=2] b
    ):

    """
    Calculate the discrete convolution of a and b
    """
    cdef:
        Py_ssize_t N1, N2, N
        int ii, jj, kk
        Py_ssize_t ii_, jj_, kk_

    N1 = a.shape[0]
    N2 = a.shape[1]
    N = (N1-1)//2

    # Output array
    cdef np.ndarray[data_t,ndim=2] yc = np.zeros([N1,N2], dtype=data_dtype)

    # yc[ii] = sum_jj a_jj b_(ii-jj)
    for ii in range(-N,N+1):
        ii_ = int_wrap(ii,N1)
        
        for jj in range(-N+int_max(0,ii), N+1+int_min(0,ii)):
            jj_ = int_wrap(jj,N1)
            kk_ = int_wrap(ii-jj,N1)
            
            for pp in range(0,N2):
              yc[ii_,pp] = yc[ii_,pp] + a[jj_,pp]*b[kk_,pp]

    return yc


cpdef df_convolve_to_real(
    np.ndarray[data_t,ndim=2] a,
    np.ndarray[data_t,ndim=2] b
    ):
    """
    Calculate the convolution of a and b.conjugate()
    """

    cdef:
        Py_ssize_t N1, N2, N
        int ii, jj, kk
        Py_ssize_t ii_, jj_, kk_

    N1 = a.shape[0]
    N2 = a.shape[1]
    N = (N1-1)//2

    # Output array
    cdef np.ndarray[data_t,ndim=2] yc = np.zeros((N+1,N2), dtype=data_dtype)

    # yc[ii] = sum_jj a_jj b_(jj-ii)^*
    for ii in range(0,N+1):
        ii_ = int_wrap(ii,N1)

        for jj in range(-N+int_max(0,ii), N+1+int_min(0,ii)):
            jj_ = int_wrap(jj,N1)
            kk_ = int_wrap(jj-ii,N1)
            
            for pp in range(0,N2):
              yc[ii_,pp] = yc[ii_,pp] + a[jj_,pp]*b[kk_,pp]

    return yc


cpdef df_triple_convolve_2d(np.ndarray[data_t, ndim=2, mode="c"] a,
                       np.ndarray[data_t, ndim=2, mode="c"] b):
    cdef:
        Py_ssize_t N1, N2, N
        Py_ssize_t ii_, jj_, kk_, ll_, pp
        int ii, jj, kk
        np.ndarray[data_t,ndim=2] c
        np.ndarray[data_t,ndim=1] bkbl

    N1 = a.shape[0]
    N2 = a.shape[1]
    N = (N1-1)//2
    c = np.zeros([N1,N2], dtype=data_dtype)

    bkbl = np.zeros(N2, dtype=data_dtype)
    for ii in range(-N,N+1):
        ii_ = int_wrap(ii,N1)

        for jj in range(-N,N+1):
          jj_ = int_wrap(jj,N1)

          bkbl.fill(0)
          for kk in range(-N+int_max(0,ii-jj), N+1+int_min(0,ii-jj)):
            kk_ = int_wrap(kk,N1)
            ll_ = int_wrap(-ii+jj+kk,N1)

            for pp in range(0,N2):
                bkbl[pp] = bkbl[pp] + a[kk_,pp]*b[ll_,pp]
        
          for pp in range(0,N2):
            c[ii_,pp] = c[ii_,pp] + a[jj_,pp]*bkbl[pp]
    return c


cpdef df_triple_convolve_1d(np.ndarray[data_t, ndim=1, mode="c"] a,
                       np.ndarray[data_t, ndim=1, mode="c"] b):
    cdef:
        Py_ssize_t N1, N2, N
        np.ndarray[data_t,ndim=1] c
        int ii, jj, kk
        Py_ssize_t ii_, jj_, kk_, ll_, pp
        data_t akbl, ajakbl

    N1 = a.shape[0]
    N = (N1-1)//2
    c = np.zeros(N1, dtype=data_dtype)

    for ii in range(-N,N+1):
        ii_ = int_wrap(ii,N1)
        
        ajakbl = (0)
        for jj in range(-N,N+1):
          jj_ = int_wrap(jj,N1)

          akbl = 0
          for kk in range(-N+int_max(0,ii-jj), N+1+int_min(0,ii-jj)):
            kk_ = int_wrap(kk,N1)
            ll_ = int_wrap(-ii+jj+kk,N1)
            akbl = akbl + a[kk_]*b[ll_]

          ajakbl = ajakbl + a[jj_]*akbl

        c[ii_] = ajakbl
    return c

