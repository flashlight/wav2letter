# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# read in the wave file
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import spectral
import scipy.misc
import wave
import sys
import struct
import numpy as np

fid = wave.open(sys.argv[1], 'r')
_, _, fs, nframes, _, _ = fid.getparams()
sig = np.array(struct.unpack_from("%dh" % nframes,
                                  fid.readframes(nframes)))
fid.close()
config = dict(fs=fs, dct=False, scale='bark', deltas=False)
extractor = spectral.Spectral(**config)
data = extractor.transform(sig)

min = np.min(data)
max = np.max(data)
print(min)
print(max)
data = data - min
data = data / (max - min)

np.savetxt(sys.argv[1] + '.mfsc', data)
scipy.misc.imsave(sys.argv[1] + '.png', data)
