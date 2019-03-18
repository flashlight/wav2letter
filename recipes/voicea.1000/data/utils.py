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
from debian.debtags import output
from curses.ascii import isdigit
from numpy import number
from awscli.customizations.emr.constants import FALSE


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

# normalize rank suffixes (1st, 2nd & 3rd)
def normalize_rank_suffix(input_str, suffix_from, suffix_to):
    output_str = input_str
    
    # find location of each pattern if any
    while True:
        pos = output_str.find(suffix_from)
        if (pos == -1):
            break
        
        # if the prior char was a digit
        # insert a 0. Example 31st -> 30 first
        if (pos > 0 and isdigit(output_str[pos - 1])):
            output_str = output_str.replace(suffix_from, "0 " + suffix_to, 1)
        else:
            output_str = output_str.replace(suffix_from, suffix_to, 1)
            
    return output_str    
    
# normalize string    
def normalize_string(input_str):
    # to lower
    output_str = input_str.lower()
    
    # replace common patterns
    output_str = output_str.replace("%", " percent")
    output_str = output_str.replace("$", " dollars")
    output_str = output_str.replace("-", " ")
    output_str = output_str.replace("_", " ")
    output_str = output_str.replace("\\", " ")
    output_str = output_str.replace("/", " ")
    
    # replace 1st - 3rd
    output_str = normalize_rank_suffix(output_str, "1st", "first")
    output_str = normalize_rank_suffix(output_str, "2nd", "second")
    output_str = normalize_rank_suffix(output_str, "3rd", "third")
        
    # convert number chunks to words
    inf = inflect.engine()
    changed = True
    while changed:
        indx = 0
        changed = False
        for indx in range(len(output_str)):
            ch = output_str[indx]
            if (isdigit(ch)):
                number_chunk = ""
                while (indx < len(output_str) and isdigit(output_str[indx])):
                    number_chunk += output_str[indx]
                    indx = indx + 1
    
                to_str = inf.number_to_words(int(number_chunk))
            
                # add a spc if the next char is not a space
                if (indx < len(output_str) and output_str[indx] != ' '):
                    to_str += " "
                    
                result = output_str.replace(number_chunk, to_str)
                output_str = result
                changed = True
                break
            
    # re-replace - with spc
    output_str = output_str.replace("-", " ")
    
    # convert accented characters
    result = unidecode.unidecode(output_str)
    if (result != output_str):
        output_str = result   
    
    # remove punctuation except '
    trans_table = output_str.maketrans("", "", 
       "[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~]:)")
    result = output_str.translate(trans_table)
    output_str = result
        
    # remove extra spaces
    ' '.join(output_str.split()) 
    
    # return normalized string
    return output_str    
    

def write_sample(sample):

    line, gender_map, idx, dst = sample
    filename, input, lbl = line.split(" ", 2)

    assert filename and input and lbl

    srcpath = os.path.dirname(filename)
    basepath = os.path.join(dst, "%09d" % idx)
    src = os.path.join(os.path.dirname(filename), input + ".flac")
    
    # check audio size
    duration = sox.file_info.duration(src)
    if (duration > 0):        
        # flac
        copytoflac(src, basepath + ".flac")
    
        # wrd
        normalized_lbl = normalize_string(lbl)
        words = normalized_lbl.strip().lower()
        with open(basepath + ".wrd", "w") as f:
            f.write(words)
    
        # ltr
        spellings = " | ".join([" ".join(w) for w in words.split()])
        with open(basepath + ".tkn", "w") as f:
            f.write(spellings)
    
        # id
        _, spkr_id, _ = srcpath.strip(os.sep).rsplit(os.sep, 2)
        with open(basepath + ".id", "w") as f:
            f.write("file_id\t{fid}".format(fid=idx))
            #f.write("\ngender\tM")
            #f.write("\nspeaker_id\t1")
