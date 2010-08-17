# _*_ coding=utf-8 _*_
#αβγδεζηθικλμνξοπρσςτυφχψω ϕϚϑ ‘’	“”  ·
#ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
#ℋℌℍℎℏℐℑℒℓℕ№℗℘ℙℚℛℜℝℤ℥ℨÅℬℰℱℳℼℽ∇
#¼½¾⅓ ⅔⅕⅖⅗⅘ ⅙⅚ ⅛

"""
Simple unit routines

"""

import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi
from scipy import random, signal
from scipy import fftpack as fftp

def dbmtow(x_dbm):
    x_w = 10**(x_dbm/10)*m_
    return x_w

p_ = 1e-12
n_ = 1e-9
u_ = 1e-6
m_ = 1e-3
k_ = 1e3
M_ = 1e6
G_ = 1e9


