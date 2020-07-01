"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to prepare recipe to train/eval model on Librispeech in wav2letter++ pipelines

Command : python3 prepare.py --data_dst [...] --model_dst [...] --wp 10000 --nbest 10

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re
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
        default="./model",
    )
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )
    parser.add_argument("--wp", help="number of word pieces", default=10000, type=int)
    parser.add_argument(
        "--nbest",
        help="number of best segmentations for each word (or numbers comma separated)",
        default="10",
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
    num_wordpieces = args.wp
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
    lexicon_words_train = sorted(word_dict["train"])
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

    # Generating decoder/*
    lm = "4-gram"
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
    os.system(
        "cat {arpa} | tr '[:upper:]' '[:lower:]' > {arpa}.lower".format(arpa=arpa_file)
    )
    lm_words = []
    with open(arpa_file + ".lower", "r") as arpa:
        for line in arpa:
            # verify if the line corresponds to unigram
            if not re.match(r"[-]*[0-9\.]+\t\S+\t*[-]*[0-9\.]*$", line):
                continue
            word = line.split("\t")[1]
            word = word.strip().lower()
            if word == "<unk>" or word == "<s>" or word == "</s>":
                continue
            assert re.match("^[a-z']+$", word), "invalid word - {w}".format(w=word)
            lm_words.append(word)

    # word -> word piece lexicon for loading targets
    print("Creating word -> word pieces lexicon...\n", flush=True)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_name)

    for nbest in args.nbest.split(","):
        nbest = int(nbest)
        lexicon_name = "librispeech-train+dev-unigram-{sz}-nbest{n}.lexicon".format(
            sz=num_wordpieces, n=nbest
        )
        lexicon_name_train = "librispeech-train-unigram-{sz}-nbest{n}.lexicon".format(
            sz=num_wordpieces, n=nbest
        )
        with open(os.path.join(am_path, lexicon_name), "w") as f_lexicon, open(
            os.path.join(am_path, lexicon_name_train), "w"
        ) as f_lexicon_train:
            for word in lexicon_words:
                wps = sp.NBestEncodeAsPieces(word, nbest)
                for wp in wps:  # the order matters for our training
                    f_lexicon.write(
                        word
                        + "\t"
                        + " ".join([w.replace("\u2581", "_") for w in wp])
                        + "\n"
                    )
                    if word in lexicon_words_train:
                        f_lexicon_train.write(
                            word
                            + "\t"
                            + " ".join([w.replace("\u2581", "_") for w in wp])
                            + "\n"
                        )
        nbest = int(nbest)
        decoder_lexicon_name = "decoder-unigram-{sz}-nbest{n}.lexicon".format(
            sz=num_wordpieces, n=nbest
        )
        with open(os.path.join(decoder_path, decoder_lexicon_name), "w") as f_lexicon:
            for word in lm_words:
                wps = sp.NBestEncodeAsPieces(word, nbest)
                for wp in wps:  # the order matters for our training
                    f_lexicon.write(
                        word
                        + "\t"
                        + " ".join([w.replace("\u2581", "_") for w in wp])
                        + "\n"
                    )
    print("Done!", flush=True)
