# encoding: utf-8
import sys
sys.path.append('..')

import numpy as np
import scipy as sp
import pylab as pl

from numpy import pi, linalg
from scipy import fftpack as fftp

from oscillate import *
from oscillate.opticalfiber import *
from oscillate.discrete import *
from oscillate.components import *
from oscillate.modulators import *
from oscillate.amplifiers import *
from oscillate.detectors import *
from oscillate.noisesources import *
from oscillate.analysis import *

# Oscillator parameters:
f_osc = 10e9                     # Design oscillation frequency (Hz)
tao = 20e-6                      # Round-trip time (s)
n_osc = round(f_osc*tao )        # Oscillation mode number
f0 = n_osc/tao                   # Exact oscillation frequency (Hz)
w0 = 2*pi*n_osc/tao              # Exact oscillation frequency (rad/s)

Gs = 1.5                         # Small signal gain
Gwidth = 8e6                     # The bandwidth of the filter
looploss = 0.33                  # Lumped loss for all components
Psat = 0.07                      # Amplifier saturation power

# Simulation parameters
Nf = 2                           # Number of harmonics
Nms = 256                        # Multi-scale discretization
Tms = tao                        # Multiscale Time window - must be round trip for full simulation
Nrt = 10000                       # Analysis round trips
Ninit = 0000                     # Initial settling round trips

# Signal definitions
m0s = MultiscaleSignal(1, Tms, Nf, w0, real=True)
mss = MultiscaleSignal(Nms, Tms, Nf, w0, real=True)

# Filter
wf = w0
filt = LorenzianFilter(wf, Gwidth, iloss=looploss)

# Noise source
noise = WhiteNoiseSource(1e-18)

# Amplifier
Ga_initial = 7.5
amp = SimpleSaturableAmplifier(Ga_initial, Psat)

# Create oscillator loop, noise is added to signal before the amplifier
loop = Loop([noise, amp, filt])

# Check small signal gain & calculate amplifier gain to give specified Gs
Gs_calc = loop.calc_ss_gain(m0s)
Ga = Ga_initial*Gs/Gs_calc
amp.set_gain(Ga)
print "Calculated amplifier gain:", Ga

# 
# The OEO delay line oscillator for signal only, we take a single point in the signal
# at each harmonic, so ignore any noise
#
def signal_spectrum(pss, loop):
    # The initial signal
    v_sig = np.ones(pss.shape, np.complex_)
    v_in = v_sig.copy()

    # The loop
    ii=0
    difference=np.inf
    while (ii<500) and (difference>1e-12):
        pss,v_sig = loop.singlepass(pss,v_in, sources=False)
        
        #Check the convergence of the signal
        difference = linalg.norm(v_in-v_sig)
        v_in = v_sig
        
        ii+=1

    Ps = pss.power(v_sig)
    print "Converged to %.4g in %d iterations to signal power P=%.3g" % (difference,ii,Ps)
    print "Loop coefficients", np.abs(np.mean(v_sig[:4], axis=1))

    return v_sig

# 
# The OEO delay line oscillator signal and noise operating spectrum
#
def noise_spectrum(mss, loop, v0_sig):
    # The initial signal
    v_sig = np.zeros(mss.shape, np.complex_)
    v_sig[:] = v0_sig

    loopsig = LoopStorage(mss, Nrt, mss.Nt)

    print "Calculating ..."

    ii=0
    while (ii<Nrt+Ninit):
        mss,v_sig = loop.singlepass(mss,v_sig)
        
        # Collect round trips
        if ii>=Ninit: loopsig.store(mss, v_sig[1])

        ii += 1

    Ps = mss.power(v_sig)
    print "Calculated noise over %d iterations, signal power P=%.3g" % (ii,Ps)
    print "Loop coefficients", np.abs(np.mean(v_sig[:4], axis=1))
    
    return loopsig
    
# 
# Plot the OEO spectrum
#
def plot_spectrum(loopsig):
    global asig
    fig1, (ax1,ax2) = pl.subplots(2, 1, num=1)

    #Extract entire signal
    dss,vs = loopsig.time_concatenated_signal()
    ax2.plot(dss.t, np.abs(vs))
    
    #Analaysis
    asig = Analysis()
    asig.set_signal(dss,vs)

    print "Noise deviation:", asig.deviation()

    f,pn = asig.phase_noise_spectrum(fstart=1e1, fend=1e6)
    f,an = asig.amplitude_noise_spectrum(fstart=1e1, fend=1e6)

    ax1.plot(f, power_to_dB(np.abs(an)**2*dss.T), 'r:', linewidth=0.5)
    ax1.plot(f, power_to_dB(np.abs(pn)**2*dss.T), 'k:', linewidth=1)

    ax1.grid(True)
    ax1.semilogx()
    ax1.set_ylim(-220,-100)
    ax1.set_xlabel(r'Frequency (Hz)')
    ax1.set_ylabel(r'Noise PSD (dB)')

    fig1.savefig('figure1_delayline_oscillator.pdf', format='pdf')

v0_sig = signal_spectrum(m0s, loop)
loopsig = noise_spectrum(mss, loop, v0_sig)
plot_spectrum(loopsig)

pl.draw()

