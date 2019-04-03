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

    subpaths = {
        "train": [
            "train-clean-100",
            "train-clean-360",
            "train-other-500"
        ],
        "dev": [
            "dev-clean",
            "dev-other"
        ],
        "test": [
            "test-clean",
            "test-other"
        ]
    }

    os.makedirs(args.dst, exist_ok=True)

    word_dict = {}
    for subpath_type in subpaths.keys():
        word_dict[subpath_type] = set()
        for subpath in subpaths[subpath_type]:
            src = os.path.join(args.src, subpath)
            assert os.path.exists(src), "Unable to find the directory - '{src}'".format(
                src=src
            )

            dst = os.path.join(args.dst, subpath + ".lst")

            sys.stdout.write("analyzing {src}...\n".format(src=src))
            sys.stdout.flush()
            transcriptfiles = utils.findtranscriptfiles(src)
            transcriptfiles.sort()

            sys.stdout.write("writing to {dst}...\n".format(dst=dst))
            sys.stdout.flush()
            with Pool(args.process) as p:
                samples = list(
                    tqdm(
                        p.imap(
                            utils.transcript_to_list,
                            transcriptfiles,
                        ),
                        total=len(transcriptfiles),
                    )
                )

            with open(dst, "w") as o:
                for sp in samples:
                    for s in sp:
                        word_dict[subpath_type].update(s[-1].split(" "))
                        s[0] = subpath + "-" + s[0]
                        o.write(" ".join(s))
                        o.write("\n")

    # create tokens dictionary
    sys.stdout.write("creating tokens list...\n")
    sys.stdout.flush()
    with open(os.path.join(args.dst, "tokens.txt"), "w") as f:
        f.write("|\n")
        f.write("'\n")
        for alphabet in range(ord("a"), ord("z") + 1):
            f.write(chr(alphabet) + "\n")

    # word -> tokens lexicon for loading targets
    sys.stdout.write("creating word -> tokens lexicon...\n")
    sys.stdout.flush()
    output_word_dict = sorted(word_dict["train"] | word_dict["dev"])
    with open(os.path.join(args.dst, "librispeech-train+dev-tokens.dict"), "w") as f:
        for w in output_word_dict:
            f.write("{word} {tokens}\n".format(word=w, tokens=" ".join(list(w))))

    sys.stdout.write("Done !\n")
