"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""
#!/usr/bin/python3

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import inflect
import sox
import re
from telnetlib import DO
import string 
import chunk
import string
import unidecode
import glob
import sys

#import pydevd
from debian.debtags import output
from curses.ascii import isdigit
from numpy import number
from awscli.customizations.emr.constants import FALSE
from threading import Lock, Thread
import normalize

def findflacfiles(dir):
    files = glob.glob(dir + "/*.flac")
    return files

def findtranscriptfiles(dir):
    files = []
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(".trans.txt"):
                files.append(os.path.join(dirpath, filename))
    return files

def copytoflac(src, dst):
    duration = sox.file_info.duration(src)
    if (duration > 0):
        sox_tfm = sox.Transformer()
        sox_tfm.set_output_format(file_type="flac", encoding="signed-integer", bits=16)
        sox_tfm.build(src, dst)
    else:
        print("File {file} has empty audio\n".format(file=src))              
        assert sox.file_info.duration(src) > 0

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
    line, idx, dst = sample
    
    id, filename, lbl = line.split(":", 2)
    assert id and filename and lbl
    
    #pydevd.settrace()
    duration = sox.file_info.duration(filename)
    if (duration > 0):
        basepath = os.path.join(dst, "%09d" % idx)
        copytoflac(filename, basepath + ".flac")
    
        # wrd
        normalized_lbl = normalize.normalize_text(lbl)
        words = normalized_lbl.strip().lower()
        with open(basepath + ".wrd", "w") as f:
            f.write(words)
    
        # ltr
        spellings = " | ".join([" ".join(w) for w in words.split()])
        with open(basepath + ".tkn", "w") as f:
            f.write(spellings)
    
        # id
        with open(basepath + ".id", "w") as f:
            f.write("file_id\t{fid}".format(fid=idx))
    else:
        sys.stdout.write("Flac {flac} is empty. Skipped!\n".format(flac=filename))