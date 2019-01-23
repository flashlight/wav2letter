"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to package original Librispeech datasets into a form readable in
wav2letter++ pipelines

Please download all the original datasets in a folder on your own
1> wget http://www.openslr.org/resources/12/dev-clean.tar.gz
2> tar xfvz dev-clean.tar.gz
# Repeat 1 and 2 for train-clean-100, train-clean-360, train-other-500,
# dev-other, test-clean, test-other

Command : python3 prepare_data.py --src [...]/LibriSpeech/ --dst [...]

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys
from multiprocessing import Pool

import utils
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument("--src", help="source directory")
    parser.add_argument("--dst", help="destination directory", default="./librispeech")
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )

    args = parser.parse_args()

    assert os.path.isdir(
        str(args.src)
    ), "Librispeech src directory not found - '{d}'".format(d=args.src)

    gender_map = utils.parse_speakers_gender("{src}/SPEAKERS.TXT".format(src=args.src))

    subpaths = {
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    }

    os.makedirs(args.dst, exist_ok=True)

    for subpath in subpaths:
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
        transcriptfiles.sort()
        sys.stdout.write("writing to {dst}...\n".format(dst=dst))
        sys.stdout.flush()

        transcripts = []
        for tf in transcriptfiles:
            with open(tf, "r") as f:
                for line in f:
                    transcripts.append(tf + " " + line.strip())

        n_samples = len(transcripts)
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
