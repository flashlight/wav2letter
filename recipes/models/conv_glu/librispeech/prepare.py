"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to prepare recipe to train/eval model on Librispeech in wav2letter++ pipelines
Please install `kenlm` on your own - https://github.com/kpu/kenlm

Command : python3 prepare.py --data_dst [...] --model_dst [...] --kenlm [...]/kenlm/

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re
from collections import defaultdict

import numpy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument(
        "--data_dst", help="data destination directory", default="./librispeech"
    )
    parser.add_argument(
        "--model_dst",
        help="model auxilary files destination directory",
        default="./conv_glu_librispeech_char",
    )
    parser.add_argument("--kenlm", help="location to installed kenlm directory")
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )

    args = parser.parse_args()
    os.system(
        "python3 {}/../../../data/librispeech/prepare.py --dst {} -p {}".format(
            os.path.dirname(os.path.abspath(__file__)), args.data_dst, args.process
        )
    )

    subpaths = {
        "train": ["train-clean-100", "train-clean-360", "train-other-500"],
        "dev": ["dev-clean", "dev-other"],
        "test": ["test-clean", "test-other"],
    }

    path_names = numpy.concatenate(list(subpaths.values()))

    lists_path = os.path.join(args.data_dst, "lists")
    am_path = os.path.join(args.model_dst, "am")
    decoder_path = os.path.join(args.model_dst, "decoder")
    os.makedirs(am_path, exist_ok=True)
    os.makedirs(decoder_path, exist_ok=True)

    # Generating am/*
    print("Generating tokens.txt for acoustic model training", flush=True)
    with open(os.path.join(am_path, "tokens.txt"), "w") as fout:
        fout.write("|\n")
        fout.write("'\n")
        for alphabet in range(ord("a"), ord("z") + 1):
            fout.write(chr(alphabet) + "\n")

    print(
        "Generating lexicon.txt (word -> tokens) for acoustic model training",
        flush=True,
    )
    word_dict = defaultdict(set)
    for key, names in subpaths.items():
        for name in names:
            with open(os.path.join(lists_path, name + ".lst"), "r") as flist:
                for line in flist:
                    transcription = line.strip().split(" ")[3:]
                    word_dict[key].update(transcription)

    lexicon_words = sorted(word_dict["train"] | word_dict["dev"])
    with open(os.path.join(am_path, "lexicon_train+dev.txt"), "w") as f:
        for word in lexicon_words:
            f.write(
                "{word}\t{tokens} |\n".format(word=word, tokens=" ".join(list(word)))
            )

    # Generating decoder/*
    lm = "4-gram"
    assert os.path.isdir(str(args.kenlm)), "kenlm directory not found - '{d}'".format(
        d=args.kenlm
    )
    print("Downloading Librispeech official LM model...\n", flush=True)
    arpa_file = os.path.join(decoder_path, lm + ".arpa")
    if not os.path.exists(arpa_file):
        os.system(
            "wget -c -O - http://www.openslr.org/resources/11/{lm}.arpa.gz | "
            "gunzip -c > {fout}".format(lm=lm, fout=arpa_file)
        )
    else:
        print("Arpa file {} exist, skip its downloading.".format(arpa_file))
    # temporary arpa file in lowercase
    print("Saving ARPA LM file in binary format ...\n", flush=True)
    os.system(
        "cat {arpa} | tr '[:upper:]' '[:lower:]' > {arpa}.tmp".format(arpa=arpa_file)
    )
    binary = os.path.join(args.kenlm, "build", "bin", "build_binary")
    os.system(
        "{bin} {farpa}.tmp {fbin}".format(
            bin=binary, farpa=arpa_file, fbin=arpa_file.replace(".arpa", ".bin")
        )
    )
    os.remove(os.path.join(arpa_file + ".tmp"))

    # prepare lexicon word -> tokens spelling
    # write words to lexicon.txt file
    lex_file = os.path.join(decoder_path, "lexicon.txt")
    print("Writing Lexicon file - {}...".format(lex_file))
    with open(lex_file, "w") as f:
        # get all the words in the arpa file
        with open(arpa_file, "r") as arpa:
            for line in arpa:
                # verify if the line corresponds to unigram
                if not re.match(r"[-]*[0-9\.]+\t\S+\t*[-]*[0-9\.]*$", line):
                    continue
                word = line.split("\t")[1]
                word = word.strip().lower()
                if word == "<unk>" or word == "<s>" or word == "</s>":
                    continue
                assert re.match("^[a-z']+$", word), "invalid word - {w}".format(w=word)
                f.write("{w}\t{s} |\n".format(w=word, s=" ".join(word)))

    print("Done!", flush=True)
