"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to prepare word-piece data for lm training

Command : python3 prepare.py --data_src [...] --model_src [...]

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

import sentencepiece as spm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LM data preparation.")
    parser.add_argument("--data_src", help="librispeech data")
    parser.add_argument("--model_src", help="model auxilary files directory")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.Load(
        os.path.join(args.model_src, "am", "librispeech-train-all-unigram-10000.model")
    )

    for name, suffix in zip(
        ["librispeech-lm-norm.txt.lower.shuffle", "dev-clean.txt", "dev-other.txt"],
        ["train", "dev-clean", "dev-other"],
    ):
        with open(os.path.join(args.data_src, "text", name), "r") as fin, open(
            os.path.join(args.model_src, "decoder/lm_wp_10k." + suffix), "w"
        ) as fout:
            for line in fin:
                result = ""
                for word in line.strip().split(" "):
                    wps = sp.NBestEncodeAsPieces(word, 1)[0]
                    result += " ".join([w.replace("\u2581", "_") for w in wps]) + " "
                fout.write("{}\n".format(result.strip()))
