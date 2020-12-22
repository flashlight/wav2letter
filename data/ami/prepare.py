"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to package original AMI dataset into a form readable in
wav2letter++ pipelines

Command : python3 prepare.py --dst [...]

Replace [...] with appropriate path
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from multiprocessing import Pool

from tqdm import tqdm
from utils import split_audio, create_limited_sup


LOG_STR = " To regenerate this file, please, remove it."

MIN_DURATION_MSEC = 50  # 50 msec
MAX_DURATION_MSEC = 30000  # 30 sec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMI Dataset creation.")
    parser.add_argument(
        "--dst",
        help="destination directory where to store data",
        default="./ami",
    )
    parser.add_argument(
        "-p",
        "--process",
        help="number of process for multiprocessing",
        default=8,
        type=int,
    )

    args = parser.parse_args()

    splits = {"train": [], "dev": [], "test": []}
    audio_path = os.path.join(args.dst, "audio")
    text_path = os.path.join(args.dst, "text")
    lists_path = os.path.join(args.dst, "lists")
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(lists_path, exist_ok=True)
    audio_http = "http://groups.inf.ed.ac.uk/ami"

    # Download the audio data
    print("Downloading the AMI audio data...", flush=True)
    cmds = []
    for split in splits.keys():
        with open(os.path.join("splits", f"split_{split}.orig")) as f:
            for line in f:
                line = line.strip()
                splits[split].append(line)
                cur_audio_path = os.path.join(audio_path, line)
                os.makedirs(cur_audio_path, exist_ok=True)
                num_meetings = 5 if line in ["EN2001a", "EN2001d", "EN2001e"] else 4
                for meetid in range(num_meetings):
                    cmds.append(
                        f"wget -nv --continue -o /dev/null -P {cur_audio_path} {audio_http}/AMICorpusMirror/amicorpus/{line}/audio/{line}.Headset-{meetid}.wav"
                    )

    for i in tqdm(range(len(cmds))):
        os.system(cmds[i])

    print("Downloading the text data ...", flush=True)
    annotver = "ami_public_manual_1.6.1.zip"
    cmd = f"wget -nv --continue -o /dev/null -P {text_path} {audio_http}/AMICorpusAnnotations/{annotver};"
    cmd = cmd + f"mkdir -p {text_path}/annotations;"
    cmd = cmd + f"unzip -q -o -d {text_path}/annotations {text_path}/{annotver} ;"
    os.system(cmd)

    print("Parsing the transcripts ...", flush=True)
    cmd = f"sh ami_xml2text.sh {text_path};"
    os.system(cmd)
    cmd = f"perl ami_split_segments.pl {text_path}/annotations/transcripts1 {text_path}/annotations/transcripts2 2>&1 > {text_path}/annotations/split_segments.log"
    os.system(cmd)

    # Prepare the audio data
    print("Segmenting audio files...", flush=True)
    with open(f"{text_path}/annotations/transcripts2") as f:
        lines = f.readlines()
    lines = [audio_path + " " + line for line in lines]
    os.makedirs(os.path.join(audio_path, "segments"), exist_ok=True)
    with Pool(args.process) as p:
        samples = list(
            tqdm(
                p.imap(split_audio, lines),
                total=len(lines),
            )
        )
    samples = [s for s in samples if s is not None]  # filter None values
    print("Wrote {} audio segment samples".format(len(samples)))

    print("Writing to list files...", flush=True)
    for split, meetings in splits.items():
        cur_samples = [s for s in samples if s[0] in meetings]
        with open(os.path.join(lists_path, f"{split}.lst"), "w") as fout:
            for sample in cur_samples:
                if (
                    float(sample[3]) > MIN_DURATION_MSEC
                    and float(sample[3]) < MAX_DURATION_MSEC
                ):
                    fout.write("\t".join(sample[1:]) + "\n")

    print("Preparing limited supervision subsets", flush=True)
    create_limited_sup(lists_path)

    print("Done!", flush=True)
