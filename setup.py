#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("boostrtrees",
                 sources=["boostrtrees.pyx", "RTreePoint2D.cpp"],
			     language="c++",
                 include_dirs=[numpy.get_include(), "/usr/local/Cellar/boost/1.65.1/include"])],
)
