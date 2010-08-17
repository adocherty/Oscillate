import os

#Use setuptools for egg installs, if possible
import setuptools

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension as CythonExtension

pkg_name = "Oscillate"
pkg_version = "0,1"

# Numpy includes
import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

inc_dirs = [numpy_include]

ext_modules = [Extension(pkg_name+".odesolve", [os.path.join(pkg_name,"odesolve.pyx")], \
                include_dirs=inc_dirs)]
ext_modules += [Extension(pkg_name+".specializedconvolve", \
                [os.path.join(pkg_name,"specializedconvolve.pyx")], include_dirs=inc_dirs)]
ext_modules += [Extension(pkg_name+".noisegeneration", \
                [os.path.join(pkg_name,"noisegeneration.pyx")], include_dirs=inc_dirs)]
ext_modules += [Extension(pkg_name+".genericcomponent", \
                [os.path.join(pkg_name,"genericcomponent.pyx")], include_dirs=inc_dirs)]

#Generated extensions

setup(
    name = 'Oscillate',
    description = "Library for modelling oscillators",
    version=pkg_version,
    author='Andrew Docherty',
    author_email='docherty@umbc.edu',
    cmdclass = {'build_ext': build_ext},
    packages = ['Oscillate'],
    zip_safe = True,
    ext_modules = ext_modules
)

