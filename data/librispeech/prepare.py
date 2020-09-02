"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to package original Librispeech datasets into a form readable in
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
from utils import find_transcript_files, transcript_to_list


LOG_STR = " To regenerate this file, please, remove it."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument(
        "--dst",
        help="destination directory where to store data",
        default="./librispeech",
    )
    parser.add_argument(
        "-p",
        "--process",
        help="number of process for multiprocessing",
        default=8,
        type=int,
    )

    args = parser.parse_args()

    subpaths = {
        "train": ["train-clean-100", "train-clean-360", "train-other-500"],
        "dev": ["dev-clean", "dev-other"],
        "test": ["test-clean", "test-other"],
    }

    subpath_names = numpy.concatenate(list(subpaths.values()))
    audio_path = os.path.join(args.dst, "audio")
    text_path = os.path.join(args.dst, "text")
    lists_path = os.path.join(args.dst, "lists")
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(lists_path, exist_ok=True)
    audio_http = "http://www.openslr.org/resources/12/"
    text_http = "http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"

    # Download the audio data
    print("Downloading the Librispeech data.", flush=True)
    for pname in subpath_names:
        if not os.path.exists(os.path.join(audio_path, "LibriSpeech", pname)):
            print("Downloading and unpacking {}...".format(pname))
            cmd = """wget -c {http}{name}.tar.gz -P {path};
                     yes n 2>/dev/null | gunzip {path}/{name}.tar.gz;
                     tar -C {path} -xf {path}/{name}.tar"""
            os.system(cmd.format(path=audio_path, http=audio_http, name=pname))
        else:
            log_str = "{} part of data exists, skip its downloading and unpacking"
            print(log_str.format(pname) + LOG_STR, flush=True)
    # Downloading text data for language model training
    if not os.path.exists(os.path.join(text_path, "librispeech-lm-norm.txt")):
        print("Downloading and unpacking text data...")
        cmd = """wget -c {http} -P {path}; yes n 2>/dev/null |
                 gunzip {path}/librispeech-lm-norm.txt.gz"""
        os.system(cmd.format(http=text_http, path=text_path))
    else:
        print("Text data exists, skip its downloading." + LOG_STR, flush=True)

    # Prepare the audio data
    print("Converting audio data into necessary format.", flush=True)
    word_dict = {}
    for subpath_type in subpaths.keys():
        word_dict[subpath_type] = set()
        for subpath in subpaths[subpath_type]:
            src = os.path.join(audio_path, "LibriSpeech", subpath)
            assert os.path.exists(src), "Unable to find the directory - '{src}'".format(
                src=src
            )

            dst_list = os.path.join(lists_path, subpath + ".lst")
            if os.path.exists(dst_list):
                print(
                    "Path {} exists, skip its generation.".format(dst_list) + LOG_STR,
                    flush=True,
                )
                continue

            print("Analyzing {src}...".format(src=src), flush=True)
            transcript_files = find_transcript_files(src)
            transcript_files.sort()

            print("Writing to {dst}...".format(dst=dst_list), flush=True)
            with Pool(args.process) as p:
                samples = list(
                    tqdm(
                        p.imap(transcript_to_list, transcript_files),
                        total=len(transcript_files),
                    )
                )

            with open(dst_list, "w") as fout:
                for sp in samples:
                    for s in sp:
                        word_dict[subpath_type].update(s[-1].split(" "))
                        s[0] = subpath + "-" + s[0]
                        fout.write(" ".join(s) + "\n")

    # Prepare text data
    current_path = os.path.join(text_path, "librispeech-lm-norm.txt.lower.shuffle")
    if not os.path.exists(current_path):
        print("Prepare text data in the necessary format.", flush=True)
        numpy.random.seed(42)
        text_data = []
        with open(os.path.join(text_path, "librispeech-lm-norm.txt"), "r") as f_text:
            for line in f_text:
                line = line.strip().lower()
                if line != "":
                    text_data.append(line)

        indices = numpy.random.permutation(numpy.arange(len(text_data)))

        with open(
            os.path.join(text_path, "librispeech-lm-norm.txt.lower.shuffle"), "w"
        ) as f:
            for index in indices:
                f.write(text_data[index] + "\n")
    else:
        print(
            "Path {} exists, skip its generation.".format(current_path) + LOG_STR,
            flush=True,
        )

    for pname in subpath_names:
        current_path = os.path.join(text_path, pname + ".txt")
        if not os.path.exists(current_path):
            with open(os.path.join(lists_path, pname + ".lst"), "r") as flist, open(
                os.path.join(text_path, pname + ".txt"), "w"
            ) as fout:
                for line in flist:
                    fout.write(" ".join(line.strip().split(" ")[3:]) + "\n")
        else:
            print(
                "Path {} exists, skip its generation.".format(current_path) + LOG_STR,
                flush=True,
            )

    print("Done!", flush=True)
