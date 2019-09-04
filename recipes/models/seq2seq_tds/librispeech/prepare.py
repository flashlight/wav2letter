"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to prepare recipe to train/eval model on Librispeech in wav2letter++ pipelines

Command : python3 prepare.py --data_dst [...] --model_dst [...]

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from collections import defaultdict

import sentencepiece as spm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument(
        "--data_dst", help="data destination directory", default="./librispeech"
    )
    parser.add_argument(
        "--model_dst",
        help="model auxilary files destination directory",
        default="./seq2seq_tds_librispeech",
    )
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
    num_wordpieces = 10000
    nbest = 10
    train_all_text = os.path.join(am_path, "train.txt")
    prefix = "librispeech-train-all-unigram-{}".format(num_wordpieces)
    prefix = os.path.join(am_path, prefix)
    vocab_name = prefix + ".vocab"
    model_name = prefix + ".model"

    # prepare data
    print("Preparing tokens and lexicon for acoustic model...\n", flush=True)
    word_dict = defaultdict(set)
    with open(train_all_text, "w") as ftext:
        for key, names in subpaths.items():
            for name in names:
                with open(os.path.join(lists_path, name + ".lst"), "r") as flist:
                    for line in flist:
                        transcription = line.strip().split(" ")[3:]
                        if key == "train":
                            ftext.write(" ".join(transcription) + "\n")
                        word_dict[key].update(transcription)
    lexicon_words = sorted(word_dict["train"] | word_dict["dev"])

    # train
    print("Computing word pieces...\n", flush=True)
    train_cmd = (
        "--input={input} --model_prefix={prefix} --vocab_size={sz}"
        " --character_coverage=1.0 --model_type=unigram"
        " --split_by_unicode_script=false".format(
            input=train_all_text, prefix=prefix, sz=num_wordpieces
        )
    )
    spm.SentencePieceTrainer.Train(train_cmd)

    # word piece dictionary
    print("Creating word piece list...\n", flush=True)
    exclude_list = {"<unk>", "<s>", "</s>"}
    with open(vocab_name.replace(".vocab", ".tokens"), "w") as fvocab_filt:
        with open(vocab_name, "r", encoding="utf-8") as fvocab:
            for line in fvocab:
                val, _ = line.strip().split("\t", 1)
                if val not in exclude_list:
                    fvocab_filt.write(val.replace("\u2581", "_") + "\n")

    # word -> word piece lexicon for loading targets
    print("Creating word -> word pieces lexicon...\n", flush=True)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_name)
    lexicon_name = "librispeech-train+dev-unigram-{sz}-nbest{n}.lexicon".format(
        sz=num_wordpieces, n=nbest
    )
    with open(os.path.join(am_path, lexicon_name), "w") as f_lexicon:
        for word in lexicon_words:
            wps = sp.NBestEncodeAsPieces(word, nbest)
            for wp in wps: # the order matters for our training 
                f_lexicon.write(
                    word
                    + "\t"
                    + " ".join([w.replace("\u2581", "_") for w in wp])
                    + "\n"
                )

    print("Done!", flush=True)
