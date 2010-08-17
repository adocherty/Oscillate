# _*_ coding=utf-8 _*_
#αβγδεζηθικλμνξοπρσςτυφχψω ϕϚϑ ‘’	“”  ·
#ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
#ℋℌℍℎℏℐℑℒℓℕ№℗℘ℙℚℛℜℝℤ℥ℨÅℬℰℱℳℼℽ∇√
#¼½¾⅓ ⅔⅕⅖⅗⅘ ⅙⅚ ⅛

"""
Module for the flexible implementation of nonlinear response functions

The harmonics are calculated by series expansions or by symbolic
generated classes.

This is the program to generate a separate module 

TODO:

 * Tests against direct realization
 * Generate more functions
 * Finite line length

"""

#import sympy as sym
import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi
from numpy import linalg as la
from scipy import fftpack as fftp

from components import Component, TransferComponent

# Debugging/profiling
import timer

import sympy as sym


def polynomial_power_index(p, m):
    """
    The algorithm is based on the following result:

    Consider a polynomial and its ``m``-th exponent::

    P(x) = sum_{i=0}^L p_i x^i
    P(x)^m = sum_{k=0}^{m L} a(m,k) x^k

    The coefficients ``a(m,k)`` can be computed using the
    J.C.P. Miller Pure Recurrence [see D.E.Knuth, Seminumerical
    Algorithms, The art of Computer Programming v.2, Addison
    Wesley, Reading, 1981;]::

    a(m,k) = 1/(k p_0) sum_{i=1}^L p_i ((m+1)i - k) a(m,k-i),
    
    with a(m,k)=0 for k<0
    """
    #Here L is the highest power of x
    #the length of p is therefore L+1
    L = len(p)-1
    a = np.zeros(m*L+1)

    #The starting value
    a[0] = p[0]**m
    
    #The recursion
    for kk in xrange(1,m*L+1):
        for ii in xrange(1,min(L,kk)+1):
            a[kk] += p[ii]*((m+1.)*ii - kk)*a[kk-ii]
        a[kk] *= 1./(kk*p[0])

    return a

def proposal_generate_expansion(Nb=3, Nc=3, Ne=3):
    """
    Try to do as much metaprogramming as possible while
    using sympy to do the dirty work

    Currently just write out the answer, and we copy and
    paste it into the function
    
    Nb: Input series highest power of x
    Nc: Ouptut series highest power of x
    Ne: Function expansion order
    """

    # Create symbols
    bstr = ["b[%d]" % ii for ii in range(-Nb,Nb+1)]
    bs = sym.symbols(bstr)

    cstr = ["c[%d]" % ii for ii in range(0,Ne)]
    cs = sym.symbols(cstr)

    x = sym.symbols('x')

    #Create polynomial
    Pb = sym.Add(*[bs[ii+Nb]*x**ii for ii in range(-Nb,Nb+1)])

    #Multiply
    Cmout = [cs[0]]+[sym.core.Number(0)]*(2*Nc)
    for ii in range(1,Ne):
        #Expand the series
        Pbn = (Pb**ii).expand()
        
        #Eliminate powers outside range of interest
        for term in Pbn.args:
            (cx,ox) = term.leadterm(x)

            #Only consider terms within the limits
            kk = int(ox)
            if abs(kk)<=Nc:
                Cmout[kk] += cs[ii]*cx

    return Cmout

def proposal_generate_expansion_with_series(e, Nb=3, Nc=3):
    """
    e:  Series coefficients
    Nb: Input series highest power of x
    Nc: Ouptut series highest power of x
    """

    # Create symbols
    bstr = ["b[%d]" % ii for ii in range(-Nb,Nb+1)]
    bs = sym.symbols(bstr)

    x = sym.symbols('x')

    #Create polynomial
    Pb = sym.Add(*[bs[ii+Nb]*x**ii for ii in range(-Nb,Nb+1)])

    #Multiply
    Cmout = [e[0]]+[sym.core.Number(0)]*(2*Nc)
    for ii in range(1,len(e)):
        #Expand the series
        Pbn = (Pb**ii).expand()
        
        #Eliminate powers outside range of interest
        for term in Pbn.args:
            (cx,ox) = term.leadterm(x)

            #Only consider terms within the limits
            kk = int(ox)
            if abs(kk)<=Nc:
                Cmout[kk] += e[ii]*cx

    return Cmout

def proposal_generate_triple_product_aab(Na=3, Nb=3, Nc=3, realc=False, cython_index=False, reverse=False):
    """
    Expand the equation:
    
        y = a a c
    
    Na: Input A series highest power of x
    Nb: Input B series highest power of x
    Nc: Input B series highest power of x
    Nd: Ouptut series highest power of x

    real : bool
        if true, return 0..Nc harmonics, if false return -Nc..Nc
    cython_index : bool
        enumerate indices as -N .. N if false or 0 .. 2*Nc+1 if true
    """
    NNa = 2*Na+1
    NNb = 2*Nb+1
    
    NNd = Nc+1 if realc else 2*Nc+1
    Nclow = 0 if realc else Nc
    
    if cython_index:
        ordered_indices_a = np.arange(-Na,Na+1) % NNa
        ordered_indices_b = np.arange(-Nb,Nb+1) % NNb
    else:
        ordered_indices_a = np.arange(-Na,Na+1)
        ordered_indices_b = np.arange(-Nb,Nb+1)

    # Create symbols
    astr = ["a[%d]" % ordered_indices_a[ii] for ii in range(NNa)]
    as_ = sym.symbols(astr)
    bstr = ["b[%d]" % ordered_indices_b[ii] for ii in range(NNb)]
    bs_ = sym.symbols(bstr)

    if reverse: bs_ = bs_[::-1]

    x = sym.symbols('x')

    #Product of a and b
    Pa = sym.Add(*[as_[ii]*x**(ii-Na) for ii in range(NNa)])
    Pb = sym.Add(*[bs_[ii]*x**(ii-Nb) for ii in range(NNb)])
    Pabc = (Pa*Pa*Pb).expand()
    
    dmout = [sym.core.Number(0)]*(NNd)
    #Eliminate powers outside range of interest
    for term in Pabc.args:
        (dx,ox) = term.leadterm(x)

        #Only consider terms within the limits
        kk = int(ox)
        if (kk>=-Nclow) and (kk<=Nc):
            dmout[kk] += dx

    return dmout


def proposal_generate_triple_product_series(Na=3, Nb=3, Nc=3, realc=False, cython_index=False, reverse=False):
    """
    Expand the equation:
    
        y = a b c
    
    Na: Input A series highest power of x
    Nb: Input B series highest power of x
    Nc: Input B series highest power of x
    Nd: Ouptut series highest power of x

    real : bool
        if true, return 0..Nc harmonics, if false return -Nc..Nc
    cython_index : bool
        enumerate indices as -N .. N if false or 0 .. 2*Nc+1 if true
    """
    NNa = 2*Na+1
    NNb = 2*Nb+1
    NNc = 2*Nc+1
    
    NNd = Nc+1 if realc else 2*Nc+1
    Nclow = 0 if realc else Nc
    
    if cython_index:
        ordered_indices_a = np.arange(-Na,Na+1) % NNa
        ordered_indices_b = np.arange(-Nb,Nb+1) % NNb
        ordered_indices_c = np.arange(-Nc,Nc+1) % NNc
    else:
        ordered_indices_a = np.arange(-Na,Na+1)
        ordered_indices_b = np.arange(-Nb,Nb+1)
        ordered_indices_c = np.arange(-Nc,Nc+1)
    
    # Create symbols
    astr = ["a[%d]" % ordered_indices_a[ii] for ii in range(NNa)]
    as_ = sym.symbols(astr)
    bstr = ["b[%d]" % ordered_indices_b[ii] for ii in range(NNb)]
    bs_ = sym.symbols(bstr)
    cstr = ["c[%d]" % ordered_indices_c[ii] for ii in range(NNc)]
    cs_ = sym.symbols(cstr)

    if reverse: cs_ = cs_[::-1]

    x = sym.symbols('x')

    #Product of a and b
    Pa = sym.Add(*[as_[ii]*x**(ii-Na) for ii in range(NNa)])
    Pb = sym.Add(*[bs_[ii]*x**(ii-Nb) for ii in range(NNb)])
    Pc = sym.Add(*[cs_[ii]*x**(ii-Nc) for ii in range(NNc)])
    Pabc = (Pa*Pb*Pc).expand()
    
    dmout = [sym.core.Number(0)]*(NNd)
    #Eliminate powers outside range of interest
    for term in Pabc.args:
        (dx,ox) = term.leadterm(x)

        #Only consider terms within the limits
        kk = int(ox)
        if (kk>=-Nclow) and (kk<=Nc):
            dmout[kk] += dx

    return dmout


def proposal_generate_product_series(Na=3, Nb=3, Nc=3, realc=False, cython_index=False):
    """
    Na: Input A series highest power of x
    Nb: Input B series highest power of x
    Nc: Ouptut series highest power of x

    real : bool
        if true, return 0..Nc harmonics, if false return -Nc..Nc
    cython_index : bool
        enumerate indices as -N .. N if false or 0 .. 2*Nc+1 if true
    """
    NNa = 2*Na+1
    NNb = 2*Nb+1

    NNc = Nc+1 if realc else 2*Nc+1
    Nclow = 0 if realc else Nc
    
    if cython_index:
        ordered_indices_a = np.arange(-Na,Na+1) % NNa
        ordered_indices_b = np.arange(-Nb,Nb+1) % NNb
    else:
        ordered_indices_a = np.arange(-Na,Na+1)
        ordered_indices_b = np.arange(-Nb,Nb+1)
    
    # Create symbols
    astr = ["a[%d]" % ordered_indices_a[ii] for ii in range(NNa)]
    as_ = sym.symbols(astr)
    bstr = ["b[%d]" % ordered_indices_b[ii] for ii in range(NNb)]
    bs_ = sym.symbols(bstr)

    x = sym.symbols('x')

    #Product of a and b
    Pa = sym.Add(*[as_[ii]*x**(ii-Na) for ii in range(NNa)])
    Pb = sym.Add(*[bs_[ii]*x**(ii-Nb) for ii in range(NNb)])
    Pab = (Pa*Pb).expand()
    
    cmout = [sym.core.Number(0)]*(NNc)
    #Eliminate powers outside range of interest
    for term in Pab.args:
        (cx,ox) = term.leadterm(x)

        #Only consider terms within the limits
        kk = int(ox)
        if (kk>=-Nclow) and (kk<=Nc):
            cmout[kk] += cx

    return cmout

def check_function_worst_case(f_sym, x_sym, n=20):
    """
    Truncating an infinite series will result in
    incorrect values.
    
    This is a (very dirty) check on the approximate
    mangitude of each term
    """

    for term in f_sym.lseries(x_sym,0):
        c,n=term.leadterm(x_sym)
        if n>20: break

        print n, abs(c)/special.binom(int(n),int(n)/2)


def calculate_series_coefficients(f_sym, x_sym, n=6, dtype=complex):
    cs = []
    s = f_sym.nseries(x_sym,0,n)
    print s
    for ii in range(n):
        if ii==0:
            c,kk = s.leadterm(x_sym)
            if kk>0: c=0
        else:
            c = s.coeff(x_sym**ii)

        if c:
            cs.append(dtype(c))
        else:
            cs.append(0)
    return cs


if __name__=="__main__":
    import textwrap, progressbar


if 1:
    # Generate triple product expansions
    Nbrange = range(1,5)

    f = open('_generated_tripleproduct_exp.pyx', 'w')
    f.write("# Note: This is an automatically generated source file.\n\n")
    f.write("import numpy as np\n")
    f.write("cimport numpy as np\n\n")

    print "Generating triple product expansions..."
    pbwidgets = [progressbar.Percentage(), ' ', progressbar.Bar('#','[',']'), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=pbwidgets, maxval=len(Nbrange)).start()

    pyx_function_header = """
cpdef %(a_dtype_t)s direct_triple_product_abc%(ii)d(%(a_dtype_t)s a,
        %(a_dtype_t)s b, %(a_dtype_t)s c):

    cdef:
        Py_ssize_t N1, N2, ii
        %(a_dtype_t)s x

    N1 = a.shape[0]
    N2 = a.shape[1]
    x = np.zeros((N1,N2), dtype=%(dtype)s)
    
    for ii in range(N2):
"""

    for ii in Nbrange:
      functypes = {"ii":ii, "a_dtype_t": "np.ndarray[np.complex128_t, ndim=2]", "dtype":"np.complex128"}
      
      f.write(pyx_function_header % functypes)

      foutput = proposal_generate_triple_product_series(ii,ii,ii,False,True,True)
      for jj in range(len(foutput)):

          foutjj = foutput[jj].__str__().replace("]",",ii]")
          lines = textwrap.wrap(foutjj, width=80)

          f.write("       x[%d,ii] = (" % jj)
          for l in lines[:-1]:
            f.write("%s \\\n        " % l)
          f.write("%s)\n" % lines[-1])
      f.write("    return x\n\n")

      pbar.update(ii-Nbrange[0])

    f.close()
    pbar.finish()

if 0:
    # Generate triple product expansions - PYTHON CODE
    Nbrange = range(1,5)

    f = open('_generated_tripleproduct.py', 'w')
    f.write("# Note: This is an automatically generated source file.\n\n")
    f.write("import numpy as np\n")

    print "Generating triple product expansions..."
    pbwidgets = [progressbar.Percentage(), ' ', progressbar.Bar('#','[',']'), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=pbwidgets, maxval=len(Nbrange)).start()

    for ii in Nbrange:
      functype = {"ii":ii, "dtype": "np.ndarray[np.complex128_t, ndim=2]"}

      f.write( "def direct_triple_product_aab%(ii)d(a, b):\n" % functype )

      f.write( "  N1 = a.shape[0]\n")
      f.write( "  N2 = a.shape[1]\n")
      f.write( "  x = np.zeros((N1,N2), dtype=np.complex128)\n")

      foutput = proposal_generate_triple_product_aab(ii,ii,ii,False,False,True)
      for jj in range(len(foutput)):
          
          lines = textwrap.wrap(foutput[jj].__str__(), width=80)
          f.write("  x[%d] = (" % jj)
          for l in lines[:-1]:
            f.write("%s \\\n    " % l)
          f.write("%s)\n" % lines[-1])
      f.write("  return x\n\n")

      pbar.update(ii-Nbrange[0])

    f.close()
    pbar.finish()


if 0:
    # Generate series product expansions
    Nbrange = range(1,3)
    Ncrange = range(1,6)
    Nerange = range(2,11)
    
    #Create source file
    f = open('_generatedseriesexpansions.pyx', 'w')
    f.write("# Note: This is an automatically generated source file.\n\n")
    f.write("import numpy as np\n\n")

    print "Generating series expansions..."
    pbwidgets = [progressbar.Percentage(), ' ', progressbar.Bar('#','[',']'), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=pbwidgets, \
                maxval=len(Nbrange)*len(Ncrange)*len(Nerange)).start()

    for ii in Nbrange:
      for jj in Ncrange:
        for nn in Nerange:
          foutput = proposal_generate_expansion(Nb=ii, Nc=jj, Ne=nn).__str__()

          f.write( "def direct_series_expansion_b%dc%de%d(b,c):\n" % (ii,jj,nn) )
          f.write( "    return np.array( \\\n" )
          for l in textwrap.wrap(foutput, width=80):
            f.write("\t%s \\\n" % l)
          f.write("\t)\n\n")
          
          pbar.update(((ii-Nbrange[0])*len(Ncrange)+jj-Ncrange[0])*len(Nerange)+nn-Nerange[0])

    f.close()
    pbar.finish()


#Check
if 0:
    print proposal_generate_product_series(Na=1,Nb=1,Nc=1,realc=False,cython_index=True).__str__()
    print proposal_generate_triple_product_series(Na=1,Nb=1,Nc=1,realc=False,cython_index=True).__str__()
    
