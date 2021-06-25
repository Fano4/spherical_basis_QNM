from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

ext = Extension("mathfunctions",
        sources=["mathfunctions.pyx",
                 "C_MathFunctions/algebra.cpp",
                 "C_MathFunctions/spherical_harmonics.cpp"],
        language="c++",
        include_dirs=[np.get_include(), '/usr/local/Caskroom/miniconda/base/envs/working_env/include/python3.8/','/usr/local/Cellar/gsl/2.7/include'],
        library_dirs=['/usr/local/Cellar/gsl/2.7/lib'],
        libraries = ['gsl'])
setup(name="math_fun",
      ext_modules=cythonize(ext))
