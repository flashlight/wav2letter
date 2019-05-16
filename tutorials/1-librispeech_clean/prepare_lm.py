"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to create LM data into a form readable in wav2letter++ decoder pipeline

Command : prepare_lm.py --dst [...]
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re
import sys


lm = "3-gram"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech LM data creation.")
    parser.add_argument("--dst", help="destination directory", default="./librispeech")

    args = parser.parse_args()

    lm_dir = os.path.join(args.dst, "lm")
    os.makedirs(lm_dir, exist_ok=True)

    # download LM
    sys.stdout.write("\nDownloading Librispeech LM - {lm}...\n\n".format(lm=lm))
    sys.stdout.flush()

    arpa_file = os.path.join(lm_dir, lm + ".arpa")
    os.system(
        "wget -O - http://www.openslr.org/resources/11/{lm}.arpa.gz | "
        "gunzip  -c > {o}".format(lm=lm, o=arpa_file)
    )

    # temporary arpa file in lowercase
    sys.stdout.write("\nConverting to lowercase ...\n\n")
    sys.stdout.flush()
    os.system(
        "cat {arpa} | tr '[:upper:]' '[:lower:]' > {arpa}.tmp".format(arpa=arpa_file)
    )
    os.system("mv {arpa}.tmp {arpa}".format(arpa=arpa_file))

    # write words to lexicon.txt file
    dict_file = os.path.join(lm_dir, "lexicon.txt")
    sys.stdout.write("\nWriting Lexicon file - {d}...\n\n".format(d=dict_file))
    sys.stdout.flush()
    with open(dict_file, "w") as f:
        # get all the words in the arpa file
        with open(arpa_file, "r") as arpa:
            for line in arpa:
                # verify if the line corresponds to unigram
                if not re.match(r"[-]*[0-9\.]+\t\S+\t*[-]*[0-9\.]*$", line):
                    continue
                word = line.split("\t")[1]
                word = word.strip().lower()
                if word == "<unk>" or word == "<s>" or word == "</s>":
                    continue
                assert re.match("^[a-z']+$", word), "invalid word - {w}".format(w=word)
                f.write("{w}\t{s} |\n".format(w=word, s=" ".join(word)))

    sys.stdout.write("Done !\n")
