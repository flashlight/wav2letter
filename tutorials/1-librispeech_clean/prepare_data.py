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

import sox


def findtranscriptfiles(dir):
    files = []
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(".trans.txt"):
                files.append(os.path.join(dirpath, filename))
    return files


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

    lists_dst = os.path.join(args.dst, "lists")
    os.makedirs(lists_dst, exist_ok=True)

    am_dst = os.path.join(args.dst, "am")
    os.makedirs(am_dst, exist_ok=True)

    train_dev_words = {}

    for subpath in subpaths:
        src = os.path.join(args.src, subpath)

        transcripts = []
        assert os.path.exists(src), "Unable to find the directory - '{src}'".format(
            src=src
        )
        dst = os.path.join(lists_dst, subpath + ".lst")
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
        with open(dst, "w") as f:
            for n in range(n_samples):
                filename, handle, lbl = transcripts[n].split(" ", 2)

                assert filename and handle and lbl
                writeline = []
                writeline.append(subpath + "-" + handle)  # sampleid
                audio_path = os.path.join(os.path.dirname(filename), handle + ".flac")
                writeline.append(audio_path)
                writeline.append(str(sox.file_info.duration(audio_path)))  # length
                transcript = lbl.strip().lower()
                writeline.append(transcript)

                f.write("\t".join(writeline) + "\n")

                if "train" in subpath or "dev" in subpath:
                    for w in transcript.split():
                        train_dev_words[w] = True

    # create tokens dictionary
    tkn_file = os.path.join(am_dst, "tokens.txt")
    sys.stdout.write("creating tokens file {t}...\n".format(t=tkn_file))
    sys.stdout.flush()
    with open(tkn_file, "w") as f:
        f.write("|\n")
        f.write("'\n")
        for alphabet in range(ord("a"), ord("z") + 1):
            f.write(chr(alphabet) + "\n")

    # create leixcon
    lexicon_file = os.path.join(am_dst, "lexicon.txt")
    sys.stdout.write("creating train lexicon file {t}...\n".format(t=lexicon_file))
    sys.stdout.flush()
    with open(lexicon_file, "w") as f:
        for w in train_dev_words.keys():
            f.write(w)
            f.write("\t")
            f.write(" ".join(w))
            f.write(" |\n")
    sys.stdout.write("Done !\n")
