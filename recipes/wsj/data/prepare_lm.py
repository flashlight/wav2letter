"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to create LM data into a form readable in wav2letter++ decoder pipeline
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import gzip
import os
import re
import subprocess
import sys

import utils


lmdirs = [
    "13_32.1/wsj1/doc/lng_modl/lm_train/np_data/87",
    "13_32.1/wsj1/doc/lng_modl/lm_train/np_data/88",
    "13_32.1/wsj1/doc/lng_modl/lm_train/np_data/89",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSJ LM data creation.")
    parser.add_argument("--wsj1", help="top level directory containing all WSJ1 discs")
    parser.add_argument("--dst", help="destination directory", default="./wsj")
    parser.add_argument("--cmudict", help="CMU dictionary")
    parser.add_argument("--wsjdict", help="WSJ *audio* dictionary")
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )

    args = parser.parse_args()
    assert os.path.isdir(str(args.wsj1)), "WSJ1 directory not found - '{d}'".format(
        d=args.wsj1
    )

    dict = {}
    ignore = {}

    okdict = utils.processdict(args.cmudict)
    notword = {}  # not a word warning

    # Form dictionary and save LM training data
    data_file = os.path.join(args.dst, "data.txt")
    with open(data_file, "w") as training_data:
        for dir in lmdirs:
            dir = os.path.join(args.wsj1, dir)
            for filename in os.listdir(dir):
                if not filename.endswith(".z"):
                    continue

                # Get text from zip files
                filename = os.path.join(dir, filename)
                process = subprocess.Popen(["zcat", filename], stdout=subprocess.PIPE)
                out, _ = process.communicate()
                text = out.decode("utf-8")
                text = re.sub("<s[^>]+>", "<s>", text)
                text = re.sub("<s>", "{", text)
                text = re.sub("</s>", "}", text)

                lines = re.finditer(r"\{(.*?)\}", text, re.MULTILINE | re.DOTALL)
                for line in lines:
                    line = line.group(1)
                    line = line.lower().strip()
                    line = re.sub(" +", " ", line)

                    sentence = []
                    for raw_word in line.split():
                        word, spelling = utils.preprocess(raw_word)
                        if len(word) > 0:
                            sentence.append(word)
                            if len(spelling) > 0 and re.match(r"[a-z']+", spelling):
                                if word not in dict:
                                    dict[word] = {"cnt": 0, "spelling": spelling}
                                dict[word]["cnt"] += 1
                            else:
                                if word not in ignore:
                                    ignore[word] = True
                                    sys.stdout.write(
                                        "\n$ ignoring word {}".format(spelling)
                                    )
                    training_data.write(" ".join(sentence) + "\n")

    # Sort words according to frequency and save lexicon
    dict = sorted(dict.items(), key=lambda kv: kv[1]["cnt"], reverse=True)
    dict_file = os.path.join(args.dst, "lexicon.txt")
    with open(dict_file, "w") as lexicon:
        for kv in dict:
            lexicon.write("{w}\t{s} |\n".format(w=kv[0], s=" ".join(kv[1]["spelling"])))

    sys.stdout.write("Done !\n")
