"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to prepare recipe to train/eval model on Librispeech in wav2letter++ pipelines

Command : python3 prepare.py --data_dst [...] --model_dst [...] --kenlm [...]/kenlm/

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys
from collections import defaultdict

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(FILE_DIR, "../utilities"))
from utils import convert_words_to_letters_asg_rep2


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

    lists_path = os.path.join(args.data_dst, "lists")
    am_path = os.path.join(args.model_dst, "am")
    decoder_path = os.path.join(args.model_dst, "decoder")
    os.makedirs(am_path, exist_ok=True)
    os.makedirs(decoder_path, exist_ok=True)

    # Generating am/*
    print("Generating tokens.lst for acoustic model training", flush=True)
    with open(os.path.join(am_path, "tokens.lst"), "w") as fout:
        fout.write("|\n")
        fout.write("'\n")
        for alphabet in range(ord("a"), ord("z") + 1):
            fout.write(chr(alphabet) + "\n")

    print(
        "Generating lexicon.lst (word -> tokens) for acoustic model training",
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
    with open(os.path.join(am_path, "lexicon_train+dev.lst"), "w") as f:
        for word in lexicon_words:
            f.write(
                "{word}\t{tokens} |\n".format(word=word, tokens=" ".join(list(word)))
            )

    # Prepare data for char lm training/evaluation
    if os.path.exists(os.path.join(decoder_path, "char_lm_data.train")):
        print(
            "Skip generation of {}. Please remove the file to regenerate it".format(
                os.path.join(decoder_path, "char_lm_data.train")
            )
        )
    else:
        convert_words_to_letters_asg_rep2(
            os.path.join(args.data_dst, "text/librispeech-lm-norm.txt.lower.shuffle"),
            os.path.join(decoder_path, "char_lm_data.train"),
        )
    convert_words_to_letters_asg_rep2(
        os.path.join(args.data_dst, "text/dev-clean.txt"),
        os.path.join(decoder_path, "char_lm_data.dev-clean"),
    )
    convert_words_to_letters_asg_rep2(
        os.path.join(args.data_dst, "text/dev-other.txt"),
        os.path.join(decoder_path, "char_lm_data.dev-other"),
    )

    # Download official 4gram model and its lexicon
    cmd = [
        "python3 {}/../../utilities/prepare_librispeech_official_lm.py",
        "--dst {}",
        "--kenlm {}",
    ]
    os.system(
        " ".join(cmd).format(
            os.path.dirname(os.path.abspath(__file__)), decoder_path, args.kenlm
        )
    )
    additional_set = {
        "bennydeck",
        "fibi",
        "moling",
        "balvastro",
        "hennerberg",
        "ambrosch",
        "quilter's",
        "yokul",
        "recuperations",
        "dowle",
        "buzzer's",
        "tarrinzeau",
        "bozzle's",
        "riverlike",
        "vendhya",
        "sprucewood",
        "macklewain",
        "macklewain's",
        "khosala",
        "derivatively",
        "gardar",
        "untrussing",
        "rathskellers",
        "telemetering",
        "drouet's",
        "sneffels",
        "glenarvan's",
        "congal's",
        "d'avrigny",
        "rangitata",
        "wahiti",
        "presty",
        "quinci",
        "troke",
        "westmere",
        "saknussemm",
        "dhourra",
        "irolg",
        "bozzle",
        "boolooroo",
        "collander",
        "finnacta",
        "canyou",
        "myrdals",
        "shimerdas",
        "impara",
        "synesius's",
        "brandd",
        "bennydeck's",
        "weiser",
        "noirtier",
        "verloc",
        "shimerda",
        "sudvestr",
        "frierson's",
        "bergez",
        "gwynplaine's",
        "breadhouse",
        "mulrady",
        "shampooer",
        "ossipon",
        "shoplets",
        "delectasti",
        "herbivore",
        "lacquey's",
        "pinkies",
        "theosophies",
        "razetta",
        "magazzino",
        "yundt",
        "testbridge",
        "officinale",
        "burgoynes",
        "novatians",
        "sandyseal",
        "chaba",
        "beenie",
        "congal",
        "doma",
        "brau",
        "mainhall",
        "verloc's",
        "zingiber",
        "vinos",
        "bush'",
        "yulka",
        "bambeday",
        "darfhulva",
        "olbinett",
        "gingle",
        "nicless",
        "stupirti",
        "ossipon's",
        "skint",
        "ruggedo's",
        "tishimingo",
        "ganny",
        "delaunay's",
        "tumble's",
        "birdikins",
        "hardwigg",
        "homoiousios",
        "docetes",
        "daguerreotypist",
        "satisfier",
        "heuchera",
        "parrishes",
        "homoousios",
        "trampe",
        "bhunda",
        "brion's",
        "fjordungr",
        "hurstwood",
        "corncakes",
        "abalone's",
        "libano",
        "scheiler",
    }

    with open(os.path.join(decoder_path, "lexicon.txt"), "a") as flex:
        for word in additional_set:
            flex.write("{}\t{}\n".format(word, " ".join(list(word)) + " |"))
    os.rename(
        os.path.join(decoder_path, "lexicon.txt"),
        os.path.join(decoder_path, "lexicon.lst"),
    )

    # prepare oov and in vocabulary samples lists
    decoder_lexicon_words = []
    with open(os.path.join(decoder_path, "lexicon.lst"), "r") as flex:
        for line in flex:
            decoder_lexicon_words.append(line.strip().split("\t")[0])
    decoder_lexicon_words = set(decoder_lexicon_words)

    for list_name in ["test-clean.lst", "test-other.lst"]:
        with open(os.path.join(lists_path, list_name), "r") as flist, open(
            os.path.join(decoder_path, list_name + ".oov"), "w"
        ) as foov, open(os.path.join(decoder_path, list_name + ".inv"), "w") as finv:
            for line in flist:
                sample_words = set(line.strip().split(" ")[3:])
                if len(sample_words - decoder_lexicon_words) > 0:
                    foov.write(line)
                else:
                    finv.write(line)

    print("Done!", flush=True)
