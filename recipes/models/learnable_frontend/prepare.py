"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to prepare recipe to train/eval model on Librispeech in wav2letter++ pipelines
Please install `sph2pipe` on your own -
see https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools \
  with commands :

  wget https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz
  tar -xzf sph2pipe_v2.5.tar.gz && cd sph2pipe_v2.5
  gcc -o sph2pipe *.c -lm

Command : python3 prepare.py \
    --src [...]/timit --data_dst [...] --model_dst [...]
    --sph2pipe [...]/sph2pipe_v2.5/sph2pipe

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument(
        "--src", help="Source directory with downloaded and unzipped TIMIT data"
    )
    parser.add_argument(
        "--data_dst", help="data destination directory", default="./wsj"
    )
    parser.add_argument(
        "--model_dst", help="model auxilary files destination directory", default="./"
    )
    parser.add_argument(
        "--sph2pipe",
        help="path to sph2pipe executable",
        default="./sph2pipe_v2.5/sph2pipe",
    )
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )
    args = parser.parse_args()
    os.system(
        "python3 {}/../../data/timit/prepare.py "
        "--src {} --sph2pipe {} --dst {} -p {}".format(
            os.path.dirname(os.path.abspath(__file__)),
            args.src,
            args.sph2pipe,
            args.data_dst,
            args.process,
        )
    )

    am_path = os.path.join(args.model_dst, "am")
    os.makedirs(am_path, exist_ok=True)
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../data/timit/phones.txt"
        ),
        "r",
    ) as fin, open(os.path.join(am_path, "tokens.txt"), "w") as fout, open(
        os.path.join(am_path, "lexicon.txt"), "w"
    ) as fout_lexicon:
        for line in fin:
            if line.strip() == "":
                continue
            fout.write(line)
            for token in line.strip().split(" "):
                fout_lexicon.write("{}\t{}\n".format(token, token))

    print("Done!", flush=True)
