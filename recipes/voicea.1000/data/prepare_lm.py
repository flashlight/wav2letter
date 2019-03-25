"""
Copyright (c) Voicea, Inc. and its affiliates.
All rights reserved.

----------

Script to package original Voicea.1000 datasets into a form readable in
wav2letter++ pipelines

Script to create LM data into a form readable in wav2letter++ decoder pipeline

Please install `kenlm` on your own - https://github.com/kpu/kenlm

Command : prepare_lm.py --dst [...] --kenlm [...]/kenlm/
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re
import sys


lm = "4-gram"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voicea.1000 LM data creation.")
    parser.add_argument("--dst", help="destination directory")
    parser.add_argument("--kenlm", help="location to installed kenlm directory")
    parser.add_argument("--arpa", help="path of arpa lm")
    parser.add_argument("--lex", help="path of arpa lm")
  
    args = parser.parse_args()

    assert os.path.isdir(str(args.kenlm)), "kenlm directory not found - '{d}'".format(
        d=args.kenlm
    )

    lm_dir = os.path.join(args.dst, "lm")
    os.makedirs(lm_dir, exist_ok=True)

    # locate arpa
    arpa_file = args.arpa 

    # temporary arpa file in lowercase
    sys.stdout.write("\nSaving ARPA LM file in binary format ...\n\n")
    sys.stdout.flush()
    os.system(
        "cat {arpa} | tr '[:upper:]' '[:lower:]' > {arpa}.tmp".format(arpa=arpa_file)
    )
    binary = os.path.join(args.kenlm, "build", "bin", "build_binary")
    os.system("{bin} {i}.tmp {o}".format(bin=binary, i=arpa_file, o=arpa_file + ".bin"))
    os.remove("{arpa}.tmp".format(arpa=arpa_file))

    # write words to lexicon.txt file
    dict_file = os.path.join(lm_dir, args.lex)
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
                
                # only write valid words
                if (re.match(
                    "^[a-z']+$", word), "invalid word - {w}\n".format(w=word)):
                    f.write("{w}\t{s}\n".format(w=word, s=" ".join(word)))
                
    sys.stdout.write("Done !\n")
