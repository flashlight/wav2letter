"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to package Switchboard, Hub05 datasets into a form readable in
wav2letter++ pipelines

Command : python3 prepare.py [-h] [--src SRC] [--dst DST] [--hub5_sdir HUB5_SDIR]
               [--hub5_tdir HUB5_TDIR] [--sph2pipe SPH2PIPE] [-p PROCESS]

Replace [...] with appropriate path
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys
from multiprocessing import Pool

import numpy
from tqdm import tqdm
from utils import process_hub5_data, process_swbd_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switchboard Dataset creation.")
    parser.add_argument(
        "--src",
        help="path to directory containing Switchboard data - /path/to/LDC97S62,",
    )
    parser.add_argument(
        "--dst", help="destination directory where to store data", default="./swbd"
    )
    parser.add_argument(
        "--hub5_sdir",
        default=None,
        help="path to hub dataset containing speech data - /path/to/LDC2002S09/"
        "('<hub5_sdir>/english' must exist)",
    )
    parser.add_argument(
        "--hub5_tdir",
        default=None,
        help="path to hub dataset containing transcript data "
        " - /path/to/LDC2002T43. ('<hub5_tdir>/reference' must exist)",
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

    assert os.path.exists(args.sph2pipe), "sph2pipe not found '{d}'".format(
        d=args.sph2pipe
    )

    audio_path = os.path.join(args.dst, "audio")
    os.makedirs(audio_path, exist_ok=True)

    text_path = os.path.join(args.dst, "text")
    os.makedirs(text_path, exist_ok=True)

    lists_path = os.path.join(args.dst, "lists")
    os.makedirs(lists_path, exist_ok=True)

    misc_path = os.path.join(args.dst, "misc")
    os.makedirs(misc_path, exist_ok=True)

    # hub dataset preparation
    if args.hub5_tdir and args.hub5_sdir:
        print("Preparing Hub'05 data ...", flush=True)

        hub5_audio_path = os.path.join(audio_path, "hub05")
        os.makedirs(hub5_audio_path, exist_ok=True)

        stm = os.path.join(args.hub5_tdir, "reference", "hub5e00.english.000405.stm")
        lines = [line.strip() for line in open(stm, "r")]
        n_samples = len(lines)
        with Pool(args.process) as p:
            processed_lines = list(
                tqdm(
                    p.imap(
                        process_hub5_data,
                        zip(
                            lines,
                            numpy.arange(n_samples),
                            [args.hub5_sdir] * n_samples,
                            [hub5_audio_path] * n_samples,
                            [args.sph2pipe] * n_samples,
                        ),
                    ),
                    total=n_samples,
                )
            )

        with open(os.path.join(lists_path, "hub05-switchboard.lst"), "w") as sfile:
            sfile.write(
                "\n".join([l for l in processed_lines if l and l.startswith("sw")])
            )
        with open(os.path.join(lists_path, "hub05-callhome.lst"), "w") as cfile:
            cfile.write(
                "\n".join([l for l in processed_lines if l and l.startswith("en")])
            )
    else:
        print(
            "--hub5_tdir and/or --hub5_sdir is empty. Not preparing Hub'05 data.",
            flush=True,
        )

    print("Preparing Switchboard data ...", flush=True)

    swbd_audio_path = os.path.join(audio_path, "switchboard")
    os.makedirs(swbd_audio_path, exist_ok=True)

    swbd_trans_path = os.path.join(misc_path, "swb_ms98_transcriptions")
    if not os.path.exists(swbd_trans_path):
        os.system(
            "wget -qO-  http://www.openslr.org/resources/5/"
            "switchboard_word_alignments.tar.gz "
            "| tar xz -C {dir}".format(dir=misc_path)
        )

    # load acronyms
    acronym_dict = {}
    with open(os.path.join(sys.path[0], "acronyms_swbd.map"), "r") as f:
        for line in f:
            a, b = line.strip().split("\t")
            acronym_dict[a] = b

    data = {}
    for dirpath, _, filenames in os.walk(swbd_trans_path):
        for filename in filenames:
            if filename.endswith("-trans.text"):
                id = filename[2:6]  # Guaranteed to be id by swb manual
                if id not in data:
                    data[id] = [id, None, None, None]
                channel = filename[6]
                if channel == "A":
                    data[id][2] = os.path.join(dirpath, filename)
                if channel == "B":
                    data[id][3] = os.path.join(dirpath, filename)

    for dirpath, _, filenames in os.walk(args.src):
        for filename in filenames:
            if filename.endswith(".sph"):
                id = filename.replace("sw0", "")[:4]
                assert id in data
                data[id][1] = os.path.join(dirpath, filename)

    n_samples = len(data)
    with Pool(args.process) as p:
        processed_lines = list(
            tqdm(
                p.imap(
                    process_swbd_data,
                    zip(
                        data.values(),
                        numpy.arange(n_samples),
                        [swbd_audio_path] * n_samples,
                        [args.sph2pipe] * n_samples,
                        [acronym_dict] * n_samples,
                    ),
                ),
                total=n_samples,
            )
        )
    processed_lines_flat = [item for sublist in processed_lines for item in sublist]
    with open(os.path.join(lists_path, "switchboard.lst"), "w") as sfile:
        sfile.write("\n".join([l for l in processed_lines_flat if l]))
