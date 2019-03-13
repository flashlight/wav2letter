"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import sox


def findtranscriptfiles(dir):
    files = []
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(".trans.txt"):
                files.append(os.path.join(dirpath, filename))
    return files


def copytoflac(src, dst):
    assert sox.file_info.duration(src) > 0
    sox_tfm = sox.Transformer()
    sox_tfm.set_output_format(file_type="flac", encoding="signed-integer", bits=16)
    sox_tfm.build(src, dst)


def parse_speakers_gender(spk_file):
    ret = {}
    with open(spk_file, "r") as f:
        for line in f:
            if line.startswith(";"):
                continue
            id, gen, _ = line.split("|", 2)
            ret[id.strip()] = gen.strip()
    return ret


def write_sample(sample):

    line, gender_map, idx, dst = sample
    filename, input, lbl = line.split(" ", 2)

    assert filename and input and lbl

    srcpath = os.path.dirname(filename)

    basepath = os.path.join(dst, "%09d" % idx)

    # flac
    copytoflac(
        os.path.join(os.path.dirname(filename), input + ".flac"), basepath + ".flac"
    )

    # wrd
    words = lbl.strip().lower()
    with open(basepath + ".wrd", "w") as f:
        f.write(words)

    # ltr
    spellings = " | ".join([" ".join(w) for w in words.split()])
    with open(basepath + ".tkn", "w") as f:
        f.write(spellings)

    # id
    _, spkr_id, _ = srcpath.strip(os.sep).rsplit(os.sep, 2)
    gender = gender_map[spkr_id]
    with open(basepath + ".id", "w") as f:
        f.write("file_id\t{fid}".format(fid=idx))
        f.write("\ngender\t{g}".format(g=gender))
        f.write("\nspeaker_id\t{g}".format(g=spkr_id))
