#
#
#
from __future__ import division
import sys, os, re

format='display'

#Check if we can set backends in matplotlib
if 'matplotlib.backends' in sys.modules:
	print "Warning: Some functionality won't work as pylab has already been imported"
	print "To avoid this import figure_publish before pylab or matplotlib.backends"
else:
	#This sets the backend so must be done before pylab import
	import matplotlib

	if format=='display':
		print "Setting matplotlib backend to gtkagg"
		matplotlib.use('gtkagg')

	elif format=='pdf':
		print "Setting matplotlib backend to pdf"
		matplotlib.use('pdf')
	
	else: #Cairo seems to be the only (noninteractive) to support multiple outputs
		print "Setting matplotlib backend to cairo"
		matplotlib.use('cairo')

import pylab, numpy, pickle

def parse_dimension(dim):
    """
    Return string in pt, in or cm as inches
    """
    pt_per_inch = 72.27
    cm_per_inch = 2.54

    finx = dim.find('pt')
    if finx>-1:
        value = float(dim[:finx])/pt_per_inch
        return value

    finx = dim.find('in')
    if finx>-1:
        value = float(dim[:finx])
        return value

    finx = dim.find('cm')
    if finx>-1:
        value = float(dim[:finx])/cm_per_inch
        return value

    #Defualt to cms
    value = float(dim[:finx])/cm_per_inch
    return value


def setup_figure(width=None, height=None, dpi=150, ratio="golden", subplotdims=[]):
    params = {}

    golden_ratio = 2./(numpy.sqrt(5)-1.0)         # Aesthetic ratio
    if type(ratio)==float:
	    fig_ratio = ratio
    else:
	    fig_ratio = {"golden":golden_ratio,"std":4./3,"wide":16./9,"square":1.0}[ratio]

    #Set size of figure
    if width and not height:
        fig_width = parse_dimension(width)
        fig_height = fig_width/fig_ratio
    elif height and not width:
        fig_height = parse_dimension(height)
        fig_width = fig_height*fig_ratio
    else:
        fig_width = parse_dimension(width)
        fig_height = parse_dimension(height)

    #Font sizes:
    fsize_title, fsize, fsize_small, fsize_tick, fsize_legend = 9,9,8,8,8

    if len(subplotdims)<4:
        spd = [0.125,0.9,0.125,0.9]
    else:
        spd = subplotdims

    serif_fonts = ['Times New Roman','Times','serif']
    sans_serif_fonts = ['Arial', 'Helvetica', 'sans-serif']
    params.update({
        'axes.unicode_minus':False,
	    'grid.linewidth':0.5,
	    'lines.markersize':3, 'lines.markeredgewidth': 0.5, 'lines.linewidth': 0.5,
	    #'lines.markeredgewidth': 0.5,
	    #'axes.linewidth': 0.5,
	    'font.serif': serif_fonts,
	    'font.sans-serif': sans_serif_fonts,
	    #'text.usetex':False,
	    'font.family':'serif',
	    'axes.formatter.limits':[-4,4], #the cutoff before 10^x format is used!
	    'figure.dpi': dpi,
	    'font.size': fsize, 'text.fontsize': fsize,
	    'axes.labelsize': fsize_tick, 'axes.titlesize': fsize_title,
	    'xtick.labelsize': fsize_tick, 'ytick.labelsize': fsize_tick,
	    'legend.fontsize': fsize_legend,
	    'figure.figsize': [fig_width,fig_height],
        'figure.subplot.bottom':spd[2],	'figure.subplot.top':spd[3],
        'figure.subplot.left':spd[0],	'figure.subplot.right':spd[1]
        })

    pylab.rcParams.update(params)

