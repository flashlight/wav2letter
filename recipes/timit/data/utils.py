"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import sox


def copytoflac(src, dst):
    assert sox.file_info.duration(src) > 0
    sox_tfm = sox.Transformer()
    sox_tfm.set_output_format(file_type="flac", encoding="signed-integer", bits=16)
    sox_tfm.build(src, dst)


def write_sample(sample):
    src, dst, idx, phones = sample
    basepath = os.path.join(dst, "%09d" % idx)

    # flac
    copytoflac(src, basepath + ".flac")

    # phn
    targets = []
    targetfile = src.replace(".wav", ".phn")
    with open(targetfile, "r") as f:
        for line in f:
            start, end, phn = line.strip().split()
            assert start and end and phn
            assert phn in phones
            targets.append(phn)

    # phn
    with open(basepath + ".tkn", "w") as f:
        f.write(" ".join(targets))

    # id
    with open(basepath + ".id", "w") as f:
        f.write("file_id\t{fid}".format(fid=idx))
