#!/usr/bin/env python3

# adapted from wav2letter/src/feature/test/MfccTest.cpp

import itertools as it
import os
import sys

from wav2letter.feature import FeatureParams, Mfcc


if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} feature_test_data_path", file=sys.stderr)
    print("  (usually: <wav2letter_root>/src/feature/test/data)", file=sys.stderr)
    sys.exit(1)

data_path = sys.argv[1]


def load_data(filename):
    path = os.path.join(data_path, filename)
    path = os.path.abspath(path)
    with open(path) as f:
        return [float(x) for x in it.chain.from_iterable(line.split() for line in f)]


wavinput = load_data("sa1.dat")
htkfeat = load_data("sa1-mfcc.htk")

assert len(wavinput) > 0
assert len(htkfeat) > 0

params = FeatureParams()
params.sampling_freq = 16000
params.low_freq_filterbank = 0
params.high_freq_filterbank = 8000
params.num_filterbank_chans = 20
params.num_cepstral_coeffs = 13
params.use_energy = False
params.zero_mean_frame = False
params.use_power = False

mfcc = Mfcc(params)
feat = mfcc.apply(wavinput)

assert len(feat) == len(htkfeat)
assert len(feat) % 39 == 0
numframes = len(feat) // 39
featcopy = feat.copy()
for f in range(numframes):
    for i in range(1, 39):
        feat[f * 39 + i - 1] = feat[f * 39 + i]
    feat[f * 39 + 12] = featcopy[f * 39 + 0]
    feat[f * 39 + 25] = featcopy[f * 39 + 13]
    feat[f * 39 + 38] = featcopy[f * 39 + 26]
differences = [abs(x[0] - x[1]) for x in zip(feat, htkfeat)]

print(f"max_diff={max(differences)}")
print(f"avg_diff={sum(differences)/len(differences)}")
