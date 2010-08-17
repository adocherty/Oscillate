import numpy as np
import pylab as pl
from numpy import linalg as la
from scipy.integrate import ode

import pstats, cProfile

from opticalfiber import *
from discrete import *

from nlsesolvers_fast import *
from odesolve import *

dtype = np.float64

if 1:
    from scipy import random
    import _generated_tripleproduct
    from _generatedtripleproductexpansions import *
    import genericcomponent
    import convolvespecial
    import timer

    t=timer.timer()
    
    N1 = 10
    N2 = 1024
    a = np.ones((2*N1+1,N2), dtype)
    b = random.random((2*N1+1,N2)).astype(dtype)
    c = np.zeros((N1,N2), dtype)

    print ":MP:"    
    t.start("mp")
    for ii in range(100):
       y1 = direct_triple_product_abc10(a,a,b)
    t.stop("mp")

    print ":Python:"
    t.start("python")
    t.stop("python")

    print ":convolve:"
    t.start("convolve")
    for ii in range(100):
        y3 = convolvespecial.df_triple_convolve(a,b)
    t.stop("convolve")

    print y1.shape, y3.shape

    print "Diff 1-3:", linalg.norm(y1-y3)
    t.report()

#cProfile.runctx("test_nls()", globals(), locals(), "Profile.prof")
#s = pstats.Stats("Profile.prof")
#s.strip_dirs().sort_stats("time").print_stats()


