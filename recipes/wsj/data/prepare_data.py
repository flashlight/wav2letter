"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to package original WSJ datasets into a form readable in wav2letter++
pipelines

Please install `sph2pipe` on your own -
see https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools \
  with commands :

  wget https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz
  tar -xzf sph2pipe_v2.5.tar.gz && cd sph2pipe_v2.5
  gcc -o sph2pipe *.c -lm

Command : python3 prepare_data.py --wsj0 [...]/WSJ0/media \
    --wsj1 [...]/WSJ1/media --dst [...] --sph2pipe [...]/sph2pipe_v2.5/sph2pipe

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
    parser = argparse.ArgumentParser(description="WSJ Dataset creation.")
    parser.add_argument("--wsj0", help="top level directory containing all WSJ0 discs")
    parser.add_argument("--wsj1", help="top level directory containing all WSJ1 discs")
    parser.add_argument("--dst", help="destination directory", default="./wsj")
    parser.add_argument(
        "--sph2pipe",
        help="path to sph2pipe executable",
        default="./sph2pipe_v2.5/sph2pipe",
    )
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )

    args = parser.parse_args()

    assert os.path.isdir(str(args.wsj0)), "WSJ0 directory not found - '{d}'".format(
        d=args.wsj0
    )
    assert os.path.isdir(str(args.wsj1)), "WSJ1 directory not found - '{d}'".format(
        d=args.wsj1
    )
    assert args.wsj0 != args.wsj1, "WSJ0 and WSJ1 directories can't be the same"
    assert os.path.exists(args.sph2pipe), "sph2pipe not found '{d}'".format(
        d=args.sph2pipe
    )

    transcripts = {}
    utils.find_transcripts(args.wsj0, transcripts)
    utils.find_transcripts(args.wsj1, transcripts)

    sets = {}
    sets["si84"] = utils.ndx2idlist(
        args.wsj0,
        "11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx",
        transcripts,
        lambda line: None if "11_2_1:wsj0/si_tr_s/401" in line else line,
    )
    assert len(sets["si84"]) == 7138

    sets["si284"] = utils.ndx2idlist(
        args.wsj0,
        "11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx",
        transcripts,
        lambda line: None if "11_2_1:wsj0/si_tr_s/401" in line else line,
    )

    sets["si284"] = utils.ndx2idlist(
        args.wsj1,
        "13_34.1/wsj1/doc/indices/si_tr_s.ndx",
        transcripts,
        None,
        sets["si284"],
    )
    assert len(sets["si284"]) == 37416

    sets["nov92"] = utils.ndx2idlist(
        args.wsj0,
        "11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx",
        transcripts,
        lambda line: line + ".wv1",
    )
    assert len(sets["nov92"]) == 333

    sets["nov92_5k"] = utils.ndx2idlist(
        args.wsj0,
        "11-13.1/wsj0/doc/indices/test/nvp/si_et_05.ndx",
        transcripts,
        lambda line: line + ".wv1",
    )
    assert len(sets["nov92_5k"]) == 330

    sets["nov93"] = utils.ndx2idlist(
        args.wsj1,
        "13_32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx",
        transcripts,
        lambda line: line.replace("13_32_1", "13_33_1"),
    )
    assert len(sets["nov93"]) == 213

    sets["nov93_5k"] = utils.ndx2idlist(
        args.wsj1,
        "13_32.1/wsj1/doc/indices/wsj1/eval/h2_p0.ndx",
        transcripts,
        lambda line: line.replace("13_32_1", "13_33_1"),
    )
    assert len(sets["nov93_5k"]) == 215

    sets["nov93dev"] = utils.ndx2idlist(
        args.wsj1, "13_34.1/wsj1/doc/indices/h1_p0.ndx", transcripts
    )
    assert len(sets["nov93dev"]) == 503

    sets["nov93dev_5k"] = utils.ndx2idlist(
        args.wsj1, "13_34.1/wsj1/doc/indices/h2_p0.ndx", transcripts
    )
    assert len(sets["nov93dev_5k"]) == 513

    os.makedirs(args.dst, exist_ok=True)

    for set_name, samples in sets.items():
        n_samples = len(samples)
        sys.stdout.write(
            "Writing {s} with {n} samples\n".format(s=set_name, n=n_samples)
        )
        sys.stdout.flush()
        with Pool(args.process) as p:
            data_dst = os.path.join(args.dst, "data", set_name)
            os.makedirs(data_dst, exist_ok=True)
            r = list(
                tqdm(
                    p.imap(
                        utils.write_sample,
                        zip(
                            samples,
                            range(n_samples),
                            [data_dst] * n_samples,
                            [args.sph2pipe] * n_samples,
                        ),
                    ),
                    total=n_samples,
                )
            )

    # create letter dictionary
    sys.stdout.write("creating tokens list...\n")
    sys.stdout.flush()
    with open(os.path.join(args.dst, "data", "tokens.txt"), "w") as f:
        f.write("|\n")
        f.write("'\n")
        for alphabet in range(ord("a"), ord("z") + 1):
            f.write(chr(alphabet) + "\n")

    sys.stdout.write("Done !\n")
