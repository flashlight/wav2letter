"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to prepare dictionary for running experiments with Librispeech datasets in
wav2letter++ pipelines

Please run prepare_data.py first to generate all the required file lists.
Please make sure sentencepiece (https://github.com/google/sentencepiece) is installed.

Command : python3 prepare_seq2seq_dict.py --src [...] --dst [...]

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys

import sentencepiece as spm
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech dictionary creation.")
    parser.add_argument("--src", help="source directory (where *.lst files are)")
    parser.add_argument("--dst", help="destination directory", default="./librispeech")

    args = parser.parse_args()

    filelists = {
        "train": [
            "train-clean-100",
            # "train-clean-360",
            # "train-other-500"
        ],
        "dev": ["dev-clean", "dev-other"],
    }

    num_wordpieces = 5000
    nbest = 10
    prefix = "librispeech-train-all-unigram-{}".format(num_wordpieces)
    prefix = os.path.join(args.dst, prefix)
    textfile = os.path.join(args.dst, "train-all.text")
    model = prefix + ".model"
    vocab = prefix + ".vocab"

    # prepare data
    sys.stdout.write("preparing data...\n")
    sys.stdout.flush()
    train_text = utils.read_list(args.src, filelists["train"])
    dev_text = utils.read_list(args.src, filelists["dev"])

    with open(textfile, "w") as f:
        for line in train_text:
            f.write(line)
            f.write("\n")

    word_dict = set()
    for line in train_text + dev_text:
        words = line.split()
        for w in words:
            word_dict.add(w)
    word_dict = sorted(word_dict)

    # train
    sys.stdout.write("computing word pieces...\n")
    sys.stdout.flush()
    train_cmd = "--input={input} --model_prefix={prefix} --vocab_size={sz} ".format(
        input=textfile, prefix=prefix, sz=num_wordpieces
    )
    train_cmd = (
        train_cmd
        + "--character_coverage=1.0 --model_type=unigram --split_by_unicode_script=false"
    )
    spm.SentencePieceTrainer.Train(train_cmd)

    # word piece dictionary
    sys.stdout.write("creating word piece list...\n")
    exclude_list = {"<unk>", "<s>", "</s>"}
    with open(vocab + "-filtered", "w") as o:
        with open(vocab, "r") as f:
            for line in f:
                v, _ = line.strip().split("\t", 1)
                if v not in exclude_list:
                    o.write(v.replace("\u2581", "_"))
                    o.write("\n")

    # word -> word piece lexicon for loading targets
    sys.stdout.write("creating word -> word pieces lexicon...\n")
    sys.stdout.flush()
    sp = spm.SentencePieceProcessor()
    sp.Load(model)
    outfile = "librispeech-train+dev-unigram-{sz}-nbest{n}.dict".format(
        sz=num_wordpieces, n=nbest
    )
    with open(os.path.join(args.dst, outfile), "w") as f:
        for word in word_dict:
            wps = sp.NBestEncodeAsPieces(word, nbest)
            for wp in wps:
                f.write(word)
                for w in wp:
                    f.write(" " + w.replace("\u2581", "_"))
                f.write("\n")

    sys.stdout.write("Done !\n")
