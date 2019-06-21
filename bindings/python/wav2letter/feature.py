#!/usr/bin/env python3

from wav2letter._feature import *


# Not sure why this is needed. Avoids this error on FB cluster:
# Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.
try:
    import libfb.py.mkl
except ImportError:
    pass
