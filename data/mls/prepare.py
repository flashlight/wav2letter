"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to package original MLS dataset into a form readable in
wav2letter++ pipelines

Command : python3 prepare.py --indir [...] --outdir [...]

Replace [...] with appropriate path
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLS Dataset preparation.")
    parser.add_argument(
        "--indir",
        help="input directory of downloaded MLS dataset of a given language",
    )
    parser.add_argument(
        "--outdir",
        help="destination directory where to store data",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    lists_path = os.path.join(args.outdir, "lists")
    os.makedirs(lists_path, exist_ok=True)

    # Preparing the list file
    for split in ["train", "dev", "test"]:
        audio_path = os.path.join(args.indir, split, "audio")
        segments_path = os.path.join(args.indir, split, "segments.txt")
        transcripts_path = os.path.join(args.indir, split, "transcripts.txt")

        list_out_path = os.path.join(lists_path, f"{split}.lst")

        # read the segments file for audio durations
        durations = {}
        with open(segments_path) as f:
            for line in f:
                cols = line.split()
                duration_ms = (float(cols[3]) - float(cols[2])) * 1000
                durations[cols[0]] = "{:.2f}".format(duration_ms)

        with open(list_out_path, 'w') as fo:
            with open(transcripts_path) as fi:
                for line in fi:
                    handle, transcript = line.split("\t")
                    speaker, book, idx = handle.split("_")
                    audio_file = os.path.join(audio_path, speaker, book, f"{handle}.flac")
                    assert os.path.exists(audio_file)
                    fo.write(handle + "\t" + audio_file + "\t" + durations[handle] + "\t" + transcript)

    print("Done!", flush=True)
