"""
Copyright (c) Voicea, Inc. and its affiliates.
All rights reserved.

----------

Script to package original Voicea.1000 datasets into a form readable in
wav2letter++ pipelines

Command : python3 prepare_data.py --src [...]/Voicea.1000/ --dst [...]

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys
import ntpath

from multiprocessing import Pool

import utils
from tqdm import tqdm
from os.path import basename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voicea.1000 Clean Dataset creation.")
    parser.add_argument("--src", help="source directory")
    parser.add_argument("--dst", help="destination directory", default="./Voicea.1000")
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )

    args = parser.parse_args()

    assert os.path.isdir(
        str(args.src)
    ), "Voicea.1000 src directory not found - '{d}'".format(d=args.src)

    gender_map = utils.parse_speakers_gender("{src}/SPEAKERS.TXT".format(src=args.src))
    #gender_map = {}

    subpaths = {
        "train-other-500"
        #"clean-trn",
        #"clean-tst",
        #"clean-tst.stratified",
    }

    os.makedirs(args.dst, exist_ok=True)

    max_words = 110;
    
    for subpath in subpaths:
        if (subpath == "clean-trn"):
            max_words = 90
            
        src = os.path.join(args.src, subpath)
        dst = os.path.join(args.dst, "data", subpath)
        os.makedirs(dst, exist_ok=True)

        transcripts = []
        assert os.path.exists(src), "Unable to find the directory - '{src}'".format(
            src=src
        )

        sys.stdout.write("analyzing {src}...\n".format(src=src))
        sys.stdout.flush()
        transcriptfiles = utils.findtranscriptfiles(src)
        sys.stdout.write("Found {cnt} examples...\n".format(cnt=len(transcriptfiles)))
        transcriptfiles.sort()
        sys.stdout.write("writing to {dst}...\n".format(dst=dst))
        sys.stdout.flush()

        transcripts = []
        for tf in transcriptfiles:
            with open(tf, "r") as f:
                # strip path
                # get id of file by removing extension
                id = basename(tf)
                if id.endswith('.trans.txt'):
                    id = id[:-10]
                    
                for line in f:
                    # count words in transcript
                    # only add transcript less than 110 words
                    words = line.split(' ')
                    if (len(words) <= max_words):
                        #transcripts.append(tf + " " + line.strip())
                        transcripts.append(tf + " " + id + " " + line.strip())

        n_samples = len(transcripts)
        sys.stdout.write("Filtered to {cnt} examples...\n".format(cnt=n_samples))
        
        with Pool(args.process) as p:
            r = list(
                tqdm(
                    p.imap(
                        utils.write_sample,
                        zip(
                            transcripts,
                            [gender_map] * n_samples,
                            range(n_samples),
                            [dst] * n_samples,
                        ),
                    ),
                    total=n_samples,
                )
            )

    # create tokens dictionary
    sys.stdout.write("creating tokens list...\n")
    sys.stdout.flush()
    with open(os.path.join(args.dst, "tokens.txt"), "w") as f:
        f.write("|\n")
        f.write("'\n")
        for alphabet in range(ord("a"), ord("z") + 1):
            f.write(chr(alphabet) + "\n")

    sys.stdout.write("Done !\n")
