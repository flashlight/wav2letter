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

Command : python3 prepare_data.py --wsj0 [...]/WSJ0/media \
    --wsj1 [...]/WSJ1/media --data_dst [...] --model_dst [...]
    --sph2pipe [...]/sph2pipe_v2.5/sph2pipe  --kenlm [...]

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re
from collections import defaultdict

import numpy


def get_spelling(word):
    spelling = re.sub(r"\(\S+\)", "", word)  # not pronounced
    spelling = re.sub(r'[,\.:\-/&\?\!\(\)";\{\}\_#]+', "", spelling)

    if word == "'single-quote":
        spelling = spelling.replace("'", "")

    return spelling


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument("--wsj0", help="top level directory containing all WSJ0 discs")
    parser.add_argument("--wsj1", help="top level directory containing all WSJ1 discs")
    parser.add_argument(
        "--data_dst", help="data destination directory", default="./wsj"
    )
    parser.add_argument(
        "--model_dst",
        help="model auxilary files destination directory",
        default="./conv_glu_librispeech_char",
    )
    parser.add_argument(
        "--wsj1_type",
        help="if you are using larger corpus LDC94S13A, set parameter to `LDC94S13A`",
        default="LDC94S13B",
    )
    parser.add_argument(
        "--sph2pipe",
        help="path to sph2pipe executable",
        default="./sph2pipe_v2.5/sph2pipe",
    )
    parser.add_argument("--kenlm", help="location to installed kenlm directory")
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )
    args = parser.parse_args()
    os.system(
        "python3 {}/../../../data/wsj/prepare.py "
        "--wsj0 {} --wsj1 {} --sph2pipe {} --wsj1_type {} --dst {} -p {}".format(
            os.path.dirname(os.path.abspath(__file__)),
            args.wsj0,
            args.wsj1,
            args.sph2pipe,
            args.wsj1_type,
            args.data_dst,
            args.process,
        )
    )

    lists_path = os.path.join(args.data_dst, "lists")
    am_path = os.path.join(args.model_dst, "am")
    lm_data_path = os.path.join(args.data_dst, "text/lm.txt")
    decoder_path = os.path.join(args.model_dst, "decoder")
    os.makedirs(am_path, exist_ok=True)
    os.makedirs(decoder_path, exist_ok=True)

    # Generating am/*
    print("Generating tokens.txt for acoustic model training", flush=True)
    with open(os.path.join(am_path, "tokens.txt"), "w") as f_tokens:
        f_tokens.write("|\n")
        f_tokens.write("'\n")
        for alphabet in range(ord("a"), ord("z") + 1):
            f_tokens.write(chr(alphabet) + "\n")

    print(
        "Generating lexicon.txt (word -> tokens) for acoustic model training",
        flush=True,
    )
    # words used in training/eval to prepare spelling
    words_set = set()
    # words from lm data and train transcription for decoder
    lexicon_dict = defaultdict(int)

    for name in ["si284", "nov93dev"]:
        with open(os.path.join(lists_path, name + ".lst"), "r") as flist:
            for line in flist:
                transcription = line.strip().split(" ")[3:]
                words_set.update(transcription)
                if name == "si284":
                    for word in transcription:
                        lexicon_dict[word] += 1

    print(
        "Writing lexicon file - {}...".format(
            os.path.join(am_path, "lexicon_si284+nov93dev.txt")
        ),
        flush=True,
    )
    with open(os.path.join(am_path, "lexicon_si284+nov93dev.txt"), "w") as f:
        for word in words_set:
            spelling = get_spelling(word)
            assert re.match(
                r"[a-z']+", spelling
            ), "invalid spelling for word '{}'".format(word)

            f.write(
                "{word}\t{tokens} |\n".format(
                    word=word, tokens=" ".join(list(spelling))
                )
            )

    # Generating decoder/*
    # prepare lexicon word -> tokens spelling
    # write words to lexicon.txt file
    print("Generating lexicon.txt (word -> tokens) for decoding", flush=True)

    lex_file = os.path.join(decoder_path, "lexicon.txt")
    print("Writing lexicon file - {}...".format(lex_file), flush=True)
    with open(lex_file, "w") as f, open(lm_data_path, "r") as f_lm:
        for line in f_lm:
            for word in line.strip().split(" "):
                lexicon_dict[word] += 1
        sorted_indices = numpy.argsort(list(lexicon_dict.values()))[::-1]
        words = list(lexicon_dict.keys())
        for index in sorted_indices:
            spelling = get_spelling(words[index])
            if re.match("^[a-z']+$", spelling):
                f.write("{w}\t{s} |\n".format(w=words[index], s=" ".join(spelling)))
            else:
                print('Ignore word "{}" in lexicon'.format(words[index]))

    # Train 4-gram language model
    train_data = os.path.join(decoder_path, "lm+si284.txt")
    os.system(
        "cp {lm_data} {dst} && cat {trans} >> {dst}".format(
            lm_data=lm_data_path,
            dst=train_data,
            trans=os.path.join(args.data_dst, "text/si284.txt"),
        )
    )
    lmplz = os.path.join(args.kenlm, "build", "bin", "lmplz")
    binary = os.path.join(args.kenlm, "build", "bin", "build_binary")
    lm_file = os.path.join(decoder_path, "lm-4g")
    cmd = "{bin} -T /tmp -S 10G --discount_fallback -o 4 --text {file} > {lm_file}.arpa"
    os.system(cmd.format(bin=lmplz, lm_file=lm_file, file=train_data))

    os.system("{bin} {lm_file}.arpa {lm_file}.bin".format(bin=binary, lm_file=lm_file))

    print("Done!", flush=True)
