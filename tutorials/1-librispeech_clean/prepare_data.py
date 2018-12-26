"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

---------

Script to package original Mini Librispeech datasets into a form readable in
wav2letter++ pipelines

[If you haven't downloaded the datasets] Please download all the original datasets
in a folder on your own
> wget -qO- http://www.openslr.org/resources/12/train-clean-100.tar.gz | tar xvz
> wget -qO- http://www.openslr.org/resources/12/dev-clean.tar.gz | tar xvz
> wget -qO- http://www.openslr.org/resources/12/test-clean.tar.gz | tar xvz

Command : prepare_data.py --src [...]/LibriSpeech/ --dst [...]

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys


def findtranscriptfiles(dir):
    files = []
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(".trans.txt"):
                files.append(os.path.join(dirpath, filename))
    return files


def write_sample(line, idx, dst):

    filename, input, lbl = line.split(" ", 2)

    assert filename and input and lbl

    basepath = os.path.join(dst, "%09d" % idx)

    # flac
    os.system(
        "cp {src} {dst}".format(
            src=os.path.join(os.path.dirname(filename), input + ".flac"),
            dst=basepath + ".flac",
        )
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
    with open(basepath + ".id", "w") as f:
        f.write("file_id\t{fid}".format(fid=idx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument("--src", help="source directory")
    parser.add_argument("--dst", help="destination directory", default="./librispeech")

    args = parser.parse_args()

    assert os.path.isdir(
        str(args.src)
    ), "Librispeech src directory not found - '{d}'".format(d=args.src)

    subpaths = ["train-clean-100", "dev-clean", "test-clean"]

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
        transcriptfiles = findtranscriptfiles(src)
        transcriptfiles.sort()
        sys.stdout.write("writing to {dst}...\n".format(dst=dst))
        sys.stdout.flush()

        transcripts = []
        for tf in transcriptfiles:
            with open(tf, "r") as f:
                for line in f:
                    transcripts.append(tf + " " + line.strip())

        n_samples = len(transcripts)
        for n in range(n_samples):
            write_sample(transcripts[n], n, dst)

    # create tokens dictionary
    tkn_file = os.path.join(args.dst, "data", "tokens.txt")
    sys.stdout.write("creating tokens file {t}...\n".format(t=tkn_file))
    sys.stdout.flush()
    with open(tkn_file, "w") as f:
        f.write("|\n")
        f.write("'\n")
        for alphabet in range(ord("a"), ord("z") + 1):
            f.write(chr(alphabet) + "\n")

    sys.stdout.write("Done !\n")
