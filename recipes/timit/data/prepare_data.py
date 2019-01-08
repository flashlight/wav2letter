"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to package original Timit dataset into a form readable in
wav2letter++ pipelines

Command : python3 prepare_data.py --src [...] --dst [...]

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
    parser = argparse.ArgumentParser(description="Timit Dataset creation.")
    parser.add_argument("--src", help="source directory")
    parser.add_argument("--dst", help="destination directory", default="./timit")
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )

    args = parser.parse_args()

    assert os.path.isdir(str(args.src)), "Timit directory not found - '{d}'".format(
        d=args.src
    )

    curdir = os.path.dirname(__file__)

    phones = []
    sys.stdout.write("writing phone tokens...\n")
    sys.stdout.flush()

    in_phn_path = os.path.join(curdir, "phones.txt")
    out_tkn_path = os.path.join(args.dst, "data", "tokens.txt")

    with open(in_phn_path, "r") as fr:
        with open(out_tkn_path, "w") as fw:
            for line in fr:
                fw.write(line)
                phones = phones + [tkn.strip() for tkn in line.split()]
    assert len(phones) == 61

    train_path = os.path.join(args.src, "timit", "train")
    sys.stdout.write("analyzing {s}...\n".format(s=train_path))
    sys.stdout.flush()

    for ds_type in ["train", "valid", "test"]:
        sys.stdout.write("writing timit/{s}...\n".format(s=ds_type))
        sys.stdout.flush()

        dst = os.path.join(args.dst, "data", ds_type)
        os.makedirs(dst, exist_ok=True)
        list_file = os.path.join(curdir, ds_type + ".lst")
        idx = 0

        src_files = []
        with open(list_file) as f:
            for filename in f:
                src_files.append(os.path.join(args.src, "timit", filename.strip()))

        n_samples = len(src_files)
        with Pool(args.process) as p:
            r = list(
                tqdm(
                    p.imap(
                        utils.write_sample,
                        zip(
                            src_files,
                            [dst] * n_samples,
                            range(n_samples),
                            [phones] * n_samples,
                        ),
                    ),
                    total=n_samples,
                )
            )

    sys.stdout.write("Done !\n")
