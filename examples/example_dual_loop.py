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
tao2 = 2e-6                      # Round-trip time slave (s)
tao1 = tao2*10                   # Round-trip time main (s)

Gs = 1.5                         # Small signal gain
P0 = dBm_to_power(14.77)         # Input optical power
eta = 1.0                        # Modulator η
Vpi = 3.14                       # Modulator half-wave voltage
Vb = 3.14                        # Modulator bias voltage
rho = 0.8                        # Detector σ
R = 50.                          # Detector impedance
Gwidth = 8e6                     # The bandwidth of the filter
looploss = 0.33                  # Lumped loss for all components

# Loop coupling coefficients
G12=0.1; G21=0.1
G11=np.sqrt(1.-G21**2); G22=np.sqrt(1.-G12**2)

# Simulation parameters
Nf = 2                           # Number of harmonics
Tms1 = tao1                      # Multiscale Time window for loop 1
Tms2 = tao2                      # Multiscale Time window for loop 2
Nms2 = 128                       # Multi-scale discretization
Nms1 = Nms2*int(Tms1//Tms2)
Nrt = 5000                       # Analysis round trips
Ninit = 10000                    # Initial settling round trips

# Calculated properties
Vph = P0*rho*R/2                 # Detector photovoltage
n_osc_1 = round(f_osc*tao1)      # Oscillation mode number
f0 = n_osc_1/tao1                # Exact oscillation frequency (Hz)
w0 = 2*pi*n_osc_1/tao1           # Exact oscillation frequency (rad/s)
print "Coupling ratio : C=%.4g dB" % (20*np.log10(G21/G11))

# Signal definitions
m01 = MultiscaleSignal(1, Tms1, Nf, w0, real=True)
ms1 = MultiscaleSignal(Nms1, Tms1, Nf, w0, real=True)
ms2 = MultiscaleSignal(Nms2, Tms2, Nf, w0, real=True)

# Modulator
modet = ModulatorDetector(eta, Vb, Vpi, Vph, iloss=looploss**2)

# Filter
wf = w0
filt = LorenzianFilter2(wf, Gwidth)

# Additive noise
noise = WhiteNoiseSource(2*R*1e-20)

# Amplifier
Ga_initial = 7.5
amp = Amplifier(Ga_initial)

loop1 = Loop([modet, noise, amp, filt])
loop2 = Loop([modet, noise, amp, filt])

# Check small signal gain & calculate amplifier gain to give specified Gs
Gs_calc = loop1.calc_ss_gain(m01)
Ga = Ga_initial*Gs/Gs_calc
amp.set_gain(Ga)
print "Calculated amplifier gain:", Ga


# 
# Test a dual delay line oscillator
#
def noise_spectrum(ms1, ms2, loop1, loop2):
    # The initial signals
    v1_sig = np.zeros(ms1.shape, np.complex_)
    v2_sig = np.zeros(ms1.shape, np.complex_)
    v2_out = np.zeros(ms2.shape, np.complex_)
   
    loopsig1 = LoopStorage(ms1, Nrt, ms1.Nt)
    loopsig2 = LoopStorage(ms1, Nrt, ms1.Nt)

    print "Calculating ..."

    n_12 = int(ms1.T//ms2.T)
    ii=0
    while (ii<Nrt+Ninit):
        # Execute loop 1 x n_12
        for jj in range(n_12):
            #v1_out is signal from last v1 loop
            #v2_out is from last v2 loop
            ms2,v1_out = ms1.take_part(v1_sig, n_12, jj)

            #Mix
            v2_in = G21*v1_out + G22*v2_out
            v1_in = G11*v1_out + G12*v2_out

            #Store signal v1_in & v2_out
            ms1.store_part(v1_sig, v1_in, n_12, jj)
            ms1.store_part(v2_sig, v2_out, n_12, jj)

            #Run v2_in through loop 2
            ms2,v2_out = loop2.singlepass(ms2,v2_in)

        #Run v1_in through loop 1
        ms1,v1_sig = loop1.singlepass(ms1,v1_sig)

        # Collect round trips
        if ii>=Ninit:
            loopsig1.store(ms1, v1_sig[1])
            loopsig2.store(ms1, v2_sig[1])
        ii += 1

    Ps = ms1.power(v1_sig)
    print "Calculated noise over %d iterations, signal power P=%.3g" % (ii,Ps)
    print "Loop 1 coefficients", np.abs(np.mean(v1_sig[:4], axis=1))
    print "Loop 2 coefficients", np.abs(np.mean(v2_sig[:4], axis=1))
    
    return loopsig1, loopsig2


def plot_spectrum(ax, loopsig):
    asig = Analysis()
    dss,vs = loopsig.time_concatenated_signal()
    asig.set_signal(dss,vs)

    f,pn = asig.phase_noise_spectrum(fstart=1e1, fend=1e6)
    f,an = asig.amplitude_noise_spectrum(fstart=1e1, fend=1e6)

    ax.plot(f, power_to_dB(np.abs(an)**2*dss.T), 'r:', linewidth=0.5)
    ax.plot(f, power_to_dB(np.abs(pn)**2*dss.T), 'k:', linewidth=1)

    ax.grid(True)
    ax.semilogx()
    ax.set_ylim(-220,-60)

loopsig1, loopsig2 = noise_spectrum(ms1, ms2, loop1, loop2)
fig1, (ax1,ax2) = pl.subplots(2, 1, num=1)

plot_spectrum(ax1,loopsig1)
plot_spectrum(ax2,loopsig2)

ax2.set_xlabel(r'Frequency (Hz)')
ax1.set_ylabel(r'Noise PSD (dB)')

fig1.savefig('figure3_dual_loop_1.pdf', format='pdf')
pl.draw()

