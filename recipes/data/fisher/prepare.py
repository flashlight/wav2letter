"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to package original Fisher datasets into a form readable in
wav2letter++ pipelines

Command : python3 prepare.py --dst [...]

Replace [...] with appropriate path
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from multiprocessing import Pool

import numpy
from tqdm import tqdm
from utils import find_files, process_fisher_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fisher Dataset creation.")
    parser.add_argument(
        "--src",
        help="comma-separated directories containing Fisher data -"
        "/path/to/LDC2004T19,/path/to/LDC2005T19,"
        "/path/to/LDC2004S13,/path/to/LDC2005S13",
    )
    parser.add_argument(
        "--dst", help="destination directory where to store data", default="./fisher"
    )
    parser.add_argument(
        "--sph2pipe",
        help="path to sph2pipe executable",
        default="./sph2pipe_v2.5/sph2pipe",
    )
    parser.add_argument(
        "-p",
        "--process",
        help="number of process for multiprocessing",
        default=8,
        type=int,
    )

    args = parser.parse_args()
    files = find_files(args.src)

    assert len(files) == 11699, (
        "Expected to find 11699 .sph and transcript files in the Fisher "
        "data, found {}".format(len(files))
    )

    audio_path = os.path.join(args.dst, "audio")
    text_path = os.path.join(args.dst, "text")
    lists_path = os.path.join(args.dst, "lists")
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(lists_path, exist_ok=True)

    n_samples = len(files)
    with Pool(args.process) as p:
        processed_lines = list(
            tqdm(
                p.imap(
                    process_fisher_data,
                    zip(
                        files,
                        numpy.arange(n_samples),
                        [audio_path] * n_samples,
                        [args.sph2pipe] * n_samples,
                    ),
                ),
                total=n_samples,
            )
        )
    processed_lines_flat = [item for sublist in processed_lines for item in sublist]
    with open(os.path.join(lists_path, "fisher.lst"), "w") as ffile:
        ffile.write("\n".join([l for l in processed_lines_flat if l]))
