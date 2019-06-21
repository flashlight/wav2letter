#!/usr/bin/env python3

from wav2letter._criterion import *


have_torch = False
try:
    import torch

    have_torch = True
except ImportError:
    pass

if have_torch:
    from wav2letter.criterion_torch import *
