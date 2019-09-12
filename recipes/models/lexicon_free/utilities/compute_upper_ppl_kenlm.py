"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Compute upper limit on word perplexity for kenlm ngram models

Command : python3 compute_upper_ppl_kenlm.py --vocab_file [...] --kenlm_preds [...]

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import numpy
from utils import transform_asg


def compute_upper_limit_ppl_for_kenlm(known_words_file, kenlm_file):
    with open(known_words_file, "r") as f:
        known_words = set(list(map(transform_asg, f.readline().strip().split(" "))))

    with open(kenlm_file, "r") as f:
        sum_logp = 0
        sum_logp_unk = 0
        n_words = 0
        n_words_unk = 0
        n_letters = 0

        for line in f:
            if "Total" not in line:
                continue
            line = line.strip().split("\t")
            word = ""
            word_logp = 0
            for token in line:
                token_val = token.split("=")[0]
                logp = float(token.split(" ")[-1])
                if token_val == "|":
                    if word in known_words:
                        sum_logp += word_logp + numpy.log(numpy.power(10, logp))
                        n_words += 1
                    else:
                        sum_logp_unk += word_logp + numpy.log(numpy.power(10, logp))
                        n_words_unk += 1
                    word = ""
                    word_logp = 0
                elif token_val == "</s>":
                    sum_logp += numpy.log(numpy.power(10, logp))
                    n_words += 1
                else:
                    word += token_val
                    word_logp += numpy.log(numpy.power(10, logp))
                n_letters += 1
                if token_val == "</s>":
                    break
        loss_letter = -(sum_logp + sum_logp_unk) / n_letters
        ppl_word_no_unk = numpy.exp(-sum_logp / n_words)
        ppl_word_unk = numpy.exp(-sum_logp_unk / n_words_unk)
        ppl_word = numpy.exp(-(sum_logp + sum_logp_unk) / (n_words + n_words_unk))

        print(
            "Letter loss: {}, letter perplexity: {}".format(
                loss_letter, numpy.exp(loss_letter)
            )
        )
        print("Upper word perplexity for all words: {}".format(ppl_word))
        print("Upper word perplexity for unknown words: {}".format(ppl_word_unk))
        print(
            "(Reported in the paper) "
            "Upper word perplexity for known words: {}".format(ppl_word_no_unk)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upper limit on word perplexity for kenlm predictions"
    )
    parser.add_argument(
        "--vocab_file",
        help="vocabulary of known words, use file "
        "from --limit_vocab_file during word kenLM training.",
    )
    parser.add_argument(
        "--kenlm_preds", help="file with kenlm predictions after query run"
    )

    args = parser.parse_args()
    print("Evaluate file {}".format(args.kenlm_preds))
    compute_upper_limit_ppl_for_kenlm(args.vocab_file, args.kenlm_preds)
