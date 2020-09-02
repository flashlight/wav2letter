"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to package original Timit dataset into a form readable in
wav2letter++ pipelines

Please install `sph2pipe` on your own -
see https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools \
  with commands :

  wget https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz
  tar -xzf sph2pipe_v2.5.tar.gz && cd sph2pipe_v2.5
  gcc -o sph2pipe *.c -lm

Command : python3 prepare.py --src [...]/timit --dst [...] \
                  --sph2pipe [...]/sph2pipe_v2.5/sph2pipe

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from multiprocessing import Pool

import numpy
import sox
from tqdm import tqdm


def copy_to_flac(info):
    src, name, dst, idx, sph2pipe = info
    original_path = os.path.join(src, name)
    path = os.path.join(dst, "%09d" % idx) + ".flac"

    if not os.path.exists(path):
        tmp_file = os.path.join(dst, "{pid}_tmp.wav".format(pid=os.getpid()))
        os.system(
            "{sph} -f wav {i} {o}".format(sph=sph2pipe, i=original_path, o=tmp_file)
        )
        assert (
            sox.file_info.duration(tmp_file) > 0
        ), "Audio file {} duration is zero.".format(original_path)

        sox_tfm = sox.Transformer()
        sox_tfm.set_output_format(file_type="flac", encoding="signed-integer", bits=16)
        sox_tfm.build(tmp_file, path)

        os.remove(tmp_file)

    duration = sox.file_info.duration(path) * 1000  # miliseconds

    transcripts = dict()
    for target_type in [".PHN", ".WRD"]:
        targets = []
        target_file = original_path.replace(".WAV", target_type)

        with open(target_file, "r") as f:
            for line in f:
                start, end, token = line.strip().split()
                assert start and end and token, "Something wrong with {} file".format(
                    target_file
                )
                targets.append(token)
        transcripts[target_type] = " ".join(targets)

    return (name, path, duration, transcripts[".WRD"], transcripts[".PHN"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timit Dataset creation.")
    parser.add_argument(
        "--src", help="Source directory with downloaded and unzipped TIMIT data"
    )
    parser.add_argument("--dst", help="destination directory", default="./timit")
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )
    parser.add_argument(
        "--sph2pipe",
        help="path to sph2pipe executable",
        default="./sph2pipe_v2.5/sph2pipe",
    )

    args = parser.parse_args()

    assert os.path.isdir(str(args.src)), "Timit directory is not found - '{d}'".format(
        d=args.src
    )
    assert os.path.exists(args.sph2pipe), "sph2pipe not found '{d}'".format(
        d=args.sph2pipe
    )

    current_dir = os.path.dirname(__file__)
    audio_path = os.path.join(args.dst, "audio")
    text_path = os.path.join(args.dst, "text")
    lists_path = os.path.join(args.dst, "lists")
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(lists_path, exist_ok=True)

    # read phone tokens
    phones = []
    in_phn_path = os.path.join(current_dir, "phones.txt")
    with open(in_phn_path, "r") as f_phones:
        phones = [[tkn.strip() for tkn in line.split()] for line in f_phones]
    phones = set(numpy.concatenate(phones))
    assert (
        len(phones) == 61
    ), "Wrong number of phones, should be 61 instrad of {}".format(len(phones))

    assert os.path.exists(os.path.join(args.src, "timit")) or os.path.exists(
        os.path.join(args.src, "TIMIT")
    ), "TIMIT data are corrupted, there is no TIMIT or timit subdirectory"
    upper_case = True if os.path.exists(os.path.join(args.src, "TIMIT")) else False

    def process_path(path, upper_case):
        return path.upper() if upper_case else path

    # prepare audio, text and lists
    for ds_type in ["train", "valid", "test"]:
        print("Writing TIMIT {} data part".format(ds_type), flush=True)
        data_list = os.path.join(current_dir, ds_type + ".lst")
        with open(data_list, "r") as f_paths:
            src_audio_files = [
                process_path(os.path.join("timit", fname.strip()), upper_case)
                for fname in f_paths
                if fname.strip() != ""
            ]

        ds_dst = os.path.join(audio_path, ds_type)
        os.makedirs(ds_dst, exist_ok=True)

        n_samples = len(src_audio_files)
        with Pool(args.process) as p:
            samples_info = list(
                tqdm(
                    p.imap(
                        copy_to_flac,
                        zip(
                            [args.src] * n_samples,
                            src_audio_files,
                            [ds_dst] * n_samples,
                            numpy.arange(n_samples),
                            [args.sph2pipe] * n_samples,
                        ),
                    ),
                    total=n_samples,
                )
            )
        with open(
            os.path.join(lists_path, "{}.phn.lst".format(ds_type)), "w"
        ) as flist, open(
            os.path.join(lists_path, "{}.lst".format(ds_type)), "w"
        ) as fwlist, open(
            os.path.join(text_path, "{}.phn.txt".format(ds_type)), "w"
        ) as ftlist, open(
            os.path.join(text_path, "{}.txt".format(ds_type)), "w"
        ) as ftwlist:
            for sample in samples_info:
                flist.write(
                    "{}\t{}\t{}\t{}\n".format(
                        sample[0], sample[1], sample[2], sample[4]
                    )
                )
                fwlist.write(
                    "{}\t{}\t{}\t{}\n".format(
                        sample[0], sample[1], sample[2], sample[3]
                    )
                )
                assert (
                    len(set(sample[4].split(" ")) - phones) == 0
                ), "Wrong phones in the transcription for sample {}".format(sample[0])
                ftlist.write("{}\n".format(sample[4]))
                ftwlist.write("{}\n".format(sample[3]))

    print("Done!", flush=True)
