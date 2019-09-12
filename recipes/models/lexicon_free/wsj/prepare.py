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
    --sph2pipe [...]/sph2pipe_v2.5/sph2pipe

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import functools
import os
import re
import sys
from collections import defaultdict

import numpy

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(FILE_DIR, "../utilities"))
from utils import convert_words_to_letters_asg_rep2

# FILE = __file__


def compare(x, y):
    # sort by counts, if counts equal then sort in lex order
    if x[1] > y[1]:
        return -1
    elif x[1] == y[1]:
        if x[0] < y[0]:
            return -1
        else:
            return 1
    else:
        return 1


def remap_words_with_same_spelling(data_dst, decoder_dst):
    words_dict = defaultdict(int)
    spellings_dict = defaultdict(set)
    spellings_appearence_dict = defaultdict(int)

    with open(os.path.join(data_dst, "lists/si284.lst"), "r") as flist:
        for line in flist:
            for word in line.strip().split(" ")[3:]:
                word = re.sub(r"\(\S+\)", "", word)  # not pronounced
                words_dict[word] += 1
                spelling = re.sub("[^a-z'.]+", "", word)
                spellings_dict[spelling].update([word])
                spellings_appearence_dict[spelling] += 1

    with open(os.path.join(data_dst, "text/lm.txt"), "r") as flm:
        for line in flm:
            for word in line.strip().split(" "):
                word = re.sub(r"\(\S+\)", "", word)  # not pronounced
                spelling = re.sub("[^a-z'.]+", "", word)
                spellings_dict[spelling].update([word])
                spellings_appearence_dict[spelling] += 1

    sorted_spellings = sorted(
        spellings_appearence_dict.items(), key=functools.cmp_to_key(compare)
    )

    special_mapping = {"al": "al-", "st": "st", "nd": "nd", "rd": "rd"}
    remap_result = dict()
    with open(os.path.join(decoder_dst, "dict-remap.txt"), "w") as fmap:
        for spelling, _ in sorted_spellings:
            words_count = {w: words_dict[w] for w in spellings_dict[spelling]}
            sorted_words = sorted(
                words_count.items(), key=functools.cmp_to_key(compare)
            )
            for word, _ in sorted_words:
                remap_result[word] = (
                    sorted_words[0][0]
                    if spelling not in special_mapping
                    else special_mapping[spelling]
                )
                fmap.write("{} {}\n".format(word, remap_result[word]))
    return remap_result


def get_spelling(word):
    spelling = re.sub(r"\(\S+\)", "", word)  # not pronounced
    spelling = re.sub("[^a-z'.]+", "", spelling)

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
    print("Generating tokens.lst for acoustic model training", flush=True)
    with open(os.path.join(am_path, "tokens.lst"), "w") as f_tokens:
        f_tokens.write("|\n")
        f_tokens.write("'\n")
        f_tokens.write(".\n")
        for alphabet in range(ord("a"), ord("z") + 1):
            f_tokens.write(chr(alphabet) + "\n")

    print(
        "Generating lexicon.lst (word -> tokens) for acoustic model training",
        flush=True,
    )

    # generating remapping for words:
    # among words with the same spelling take the most frequent word
    # use only this word in the lexicon
    # do this substitution for the dev during
    # acoustic model training for WER computation
    remap_dict = remap_words_with_same_spelling(args.data_dst, decoder_path)
    with open(os.path.join(lists_path, "si284.lst"), "r") as fin, open(
        os.path.join(am_path, "si284.lst.remap"), "w"
    ) as fout:
        for line in fin:
            line = line.strip().split(" ")
            for index in range(3, len(line)):
                word = re.sub(r"\(\S+\)", "", line[index])
                line[index] = remap_dict[word]
            fout.write(" ".join(line) + "\n")

    # words used in training/eval to prepare spelling
    words_set = set()

    for name in [
        os.path.join(am_path, "si284.lst.remap"),
        os.path.join(lists_path, "nov93dev.lst"),
    ]:
        with open(name, "r") as flist:
            for line in flist:
                transcription = line.strip().split(" ")[3:]
                words_set.update(transcription)

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
                r"[a-z'.]+", spelling
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

    lex_file = os.path.join(decoder_path, "lexicon.lst")
    print("Writing lexicon file - {}...".format(lex_file), flush=True)
    with open(lex_file, "w") as f:
        for word in numpy.unique(list(remap_dict.values())):
            if len(re.findall(r"\d", word)) > 0:
                continue
            spelling = get_spelling(word)
            if spelling != "":
                if re.match("^[a-z'.]+$", spelling):
                    f.write("{w}\t{s} |\n".format(w=word, s=" ".join(spelling)))
                else:
                    print('Ignore word "{}" in lexicon'.format(word))

    # Prepare data for char lm training/evaluation
    if os.path.exists(os.path.join(decoder_path, "char_lm_data.train")):
        print(
            "Skip generation of {}. Please remove the file to regenerate it".format(
                os.path.join(decoder_path, "char_lm_data.train")
            )
        )
    else:
        convert_words_to_letters_asg_rep2(
            os.path.join(args.data_dst, "text/lm.txt"),
            os.path.join(decoder_path, "char_lm_data.train"),
        )
    convert_words_to_letters_asg_rep2(
        os.path.join(args.data_dst, "text/nov93dev.txt"),
        os.path.join(decoder_path, "char_lm_data.nov93dev"),
    )

    with open(os.path.join(args.data_dst, "text/nov93dev.txt"), "r") as f, \
         open(os.path.join(decoder_path, "word_lm_data.nov93dev"), "w") as fout:
        for line in f:
            result = []
            for word in line.strip().split(" "):
                word = re.sub("[^a-z'.]+", "", word)
                if word != "":
                    result.append(word)
            fout.write(" ".join(result) + "\n")
    print("Done!", flush=True)
