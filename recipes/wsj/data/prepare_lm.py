"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Script to create LM data into a form readable in wav2letter++ decoder pipeline
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import gzip
import os

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

    for dir in lmdirs:
        dir = os.path.join(args.wsj1, dir)
        for filename in os.listdir(dir):
            if not filename.endswith(".z"):
                continue
            filename = os.path.join(dir, filename)
            with gzip.open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.lower()

# TODO (vineelkpratap): finish LM processing for WSJ
