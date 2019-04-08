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

#import pydevd;
import argparse
import os
import sys
import ntpath

from multiprocessing import Pool

import utils
import normalize

from tqdm import tqdm
from os.path import basename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voicea.1000 w/perturb Dataset creation.")
    parser.add_argument("--src", help="source directory")
    parser.add_argument("--dst", help="destination directory", default="./Voicea.1000")
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=30, type=int
    )

    args = parser.parse_args()

    assert os.path.isdir(
        str(args.src)
    ), "Voicea.1000 src directory not found - '{d}'".format(d=args.src)

    gender_map = utils.parse_speakers_gender("{src}/SPEAKERS.TXT".format(src=args.src))
    #gender_map = {}

    subpaths = {
        args.src + "/perturbed",
        args.src + "/clean-trn",
        #"clean-tst",
        #"clean-tst.stratified",
    }
    
    transcipt_subpaths = {
        "clean-trn",
        #"clean-tst",
        #"clean-tst.stratified",
    }

    os.makedirs(args.dst, exist_ok=True)
    max_words = 90;
    
    # extract transcripts and create transcript id dictionary
    for subpath in transcipt_subpaths:
        src = os.path.join(args.src, subpath)
        transcripts = []
        assert os.path.exists(src), "Unable to find the directory - '{src}'".format(src=src)

        sys.stdout.write("analyzing {src}...\n".format(src=src))
        sys.stdout.flush()
        transcriptfiles = utils.findtranscriptfiles(src)
        sys.stdout.write("Found {cnt} transcript files. Normalizing...\n".
                         format(cnt=len(transcriptfiles)))
        transcriptfiles.sort()
        
        transcripts = {}
        for tf in transcriptfiles:
            with open(tf, "r") as f:
                # strip path
                # get id of file by removing extension
                id = basename(tf)
                if id.endswith('.trans.txt'):
                    id = id[:-10]
                    
                for line in f:
                    # count words in transcript
                    # only add transcript less than max_words words
                    words = line.split(' ')
                    if (len(words) <= max_words):
                        #transcripts.append(tf + " " + line.strip())
                        transcripts[id] = normalize.normalize_text(line.strip())

        
        sys.stdout.write("Filtered to {cnt} examples...\n".
                         format(cnt=len(transcripts)))
    
    # create destination dir under "data"
    os.makedirs(args.dst, exist_ok=True)
    sys.stdout.write("writing to {dst}...\n".format(dst=args.dst))
    sys.stdout.flush()

    # for all subpaths
    all_flacs = []
    for subpath in subpaths:
        sys.stdout.write("Reading {dir} ...\n".format(dir=subpath))
        flacs = utils.findflacfiles(subpath)
        sys.stdout.write("Found {cnt} flac files in {dir}!\n".format( 
                         cnt=len(flacs), dir=subpath))
        #parse id
        for flac in flacs:
            filename = os.path.splitext(basename(flac))[0]
            if (filename.find("_") == -1):
                id = filename
            else:
                id, extra = filename.split("_", 1)
            
            if (id in transcripts):
                transcript = id + ":" + flac + ":" + transcripts[id]
                all_flacs.append(transcript )
    
    n_samples = len(all_flacs)        
    sys.stdout.write("Total of {cnt} flacs will be copied!\n".format(cnt=n_samples))
    
    with Pool(args.process) as p:
        r = list(
            tqdm(
                p.imap(
                    utils.write_sample,
                    zip(
                        all_flacs,
                        range(n_samples),
                        [args.dst] * n_samples,
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
