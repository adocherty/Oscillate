import os

#Use setuptools for egg installs, if possible
try:
    import setuptools
except:
    pass

from distutils.core import setup
from distutils.extension import Extension

pkg_name = "oscillate"
pkg_version = "0,1"

# Numpy includes
import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

inc_dirs = [numpy_include]
lib_dirs = []
libraries = []

ext_modules = [Extension(pkg_name+".odesolve", [os.path.join(pkg_name,"odesolve.c")], \
                include_dirs=inc_dirs, library_dirs=lib_dirs, libraries=libraries)]
ext_modules += [Extension(pkg_name+".specializedconvolve", \
                [os.path.join(pkg_name,"specializedconvolve.c")], include_dirs=inc_dirs)]
ext_modules += [Extension(pkg_name+".noisegeneration", \
                [os.path.join(pkg_name,"noisegeneration.c")], include_dirs=inc_dirs)]

#Generated extensions

setup(
    name = 'Oscillate',
    description = "Library for modelling oscillators",
    version=pkg_version,
    author='Andrew Docherty',
    author_email='docherty@umbc.edu',
    packages = ['oscillate'],
    zip_safe = True,
    ext_modules = ext_modules
)

