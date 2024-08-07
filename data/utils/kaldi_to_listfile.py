"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
----------
Script to package kaldi data directory into a form readable in
wav2letter++ pipelines

Command : python3 prepare.py --src [...] --dst [...]
Replace [...] with appropriate path

`src` directory is the path to kaldi data directory typically
prepared with `prepare_data.sh` script.

`dst` directory is the path to store (segmented) audio files and the
list file that is used by wav2letter++ pipelines to load data.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re
from multiprocessing import Pool

import sox
from tqdm import tqdm


def run_segment(item):
    uid, val = item
    infile, start_sec, end_sec, outfile = val
    sox_tfm = sox.Transformer()
    sox_tfm.set_output_format(file_type="flac", encoding="signed-integer", bits=16)
    sox_tfm.trim(start_sec, end_sec)
    sox_tfm.build(infile, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate list file from Kaldi data dir"
    )
    parser.add_argument(
        "--src",
        help="input kaldi data directory. Must contain "
        "'text', 'segments' and 'wav.scp' files",
    )
    parser.add_argument(
        "--dst",
        help="destination directory where to store data",
    )
    parser.add_argument(
        "--name", help="name of the output list file", default="data.lst"
    )
    parser.add_argument(
        "-p",
        "--process",
        help="number of process for multiprocessing",
        default=8,
        type=int,
    )

    args = parser.parse_args()

    wav_files = {}
    cache = {}
    cmds = []
    with open(f"{args.src}/wav.scp") as f:
        for line in f:
            # handles two possible cases
            # Case 1: ID followed by wav file
            # Ex: S03_U01.CH1 /path/S03_U01.CH1.wav
            # Case 2: ID followed by sox script
            # Ex: P09_S03.L sox /path/S03_P09.wav -t wav - remix 1 |
            wid, wav_handle = line.strip().split(" ", 1)
            if wav_handle in cache:
                wav_file = cache[wav_handle]
            elif wav_handle.startswith("sox"):
                hsh = re.sub("[^0-9a-zA-Z]+", "", wav_handle)
                wav_file = "/tmp/{}.wav".format(hsh)
                cmds.append(
                    wav_handle.replace(" - ", " " + wav_file + " ").replace("|", "")
                )
            else:
                wav_file = wav_handle
            wav_files[wid] = wav_file
    print("Found {} wav files".format(len(wav_files)))

    print("Running {} wav commands ...".format(len(cmds)))

    def run_command(cmd):
        os.system(cmd)

    p = Pool(args.process)
    list(
        tqdm(
            p.imap(run_command, cmds),
            total=len(cmds),
        )
    )

    transcripts = {}
    with open(f"{args.src}/text") as f:
        for line in f:
            line_split = line.strip().split()
            transcripts[line_split[0]] = " ".join(line_split[1:])
    print("Found {} transcripts".format(len(transcripts)))

    segments = {}
    with open(f"{args.src}/segments") as f:
        for line in f:
            uid, wid, start_sec, end_sec = line.strip().split(" ", 3)
            start_sec = float(start_sec)
            end_sec = float(end_sec)
            outfile = f"{args.dst}/audio/{uid}.flac"
            segments[uid] = (wav_files[wid], start_sec, end_sec, outfile)
    print("Found {} segments".format(len(segments)))

    os.makedirs(f"{args.dst}", exist_ok=True)
    os.makedirs(f"{args.dst}/audio", exist_ok=True)

    print("Creating segmented audio files ...")
    list(
        tqdm(
            p.imap(run_segment, segments.items()),
            total=len(segments),
        )
    )

    print("Writing to list file ...")
    with open(f"{args.dst}/{args.name}", "w") as fo:
        for uid, val in segments.items():
            _, start_sec, end_sec, outfile = val
            duration = "{:.2f}".format((end_sec - start_sec) * 1000)
            fo.write("\t".join([uid, outfile, duration, transcripts[uid]]) + "\n")

    print("Done!")
