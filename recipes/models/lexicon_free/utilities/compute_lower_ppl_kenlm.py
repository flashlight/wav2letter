"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Compute upper and lower limits on word perplexity for kenlm ngram models

Command : python3 compute_upper_ppl_kenlm.py --vocab_file [...] --text [...] \
            --char_model [...] --word_model [...]

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import kenlm
import numpy
from utils import EOS, prepare_vocabs, transform_asg, transform_asg_back


LOG10 = numpy.log(10)


def compute_word_logprob(model, current_state, target_word):
    word_pred = 0
    if target_word == EOS:
        stateOut = kenlm.State()
        word_pred += model.BaseScore(current_state, str(target_word), stateOut) * LOG10
    else:
        stateIn = current_state
        for token in list(target_word):
            stateOut = kenlm.State()
            word_pred += model.BaseScore(stateIn, token, stateOut) * LOG10
            stateIn = stateOut
    return word_pred


def compute_denominator(model, current_state, words):
    preds = [compute_word_logprob(model, current_state, word) for word in words]
    max_pred = numpy.max(preds)
    return max_pred + numpy.log(numpy.sum(numpy.exp(preds - max_pred)))


def compute_words_model_pdf_mass(word_model, known_words, stateIn):
    probs = []
    for word in known_words:
        stateOut = kenlm.State()
        probs.append(
            numpy.power(10, word_model.BaseScore(stateIn, str(word), stateOut))
        )
    probs_arr = numpy.array(probs)
    indices = numpy.argsort(-probs_arr)
    top = numpy.where(numpy.cumsum(probs_arr[indices]) > 0.95)[0][0]
    return [
        transform_asg(w) + "|" if w != EOS else w for w in known_words[indices[:top]]
    ]


def compute_ppl_lower_limit(
    model, word_model, sentences, known_words, known_words_original
):
    n_words = 0
    unk_n_words = 0
    ppl = 0.0
    ppl_lower = 0.0

    n_logging = len(sentences)
    for n, sentence in enumerate(sentences):
        stateIn = kenlm.State()
        word_stateIn = kenlm.State()
        model.BeginSentenceWrite(stateIn)
        word_model.BeginSentenceWrite(word_stateIn)
        current_word = ""
        word_score = 0.0
        word_state = stateIn  # state for char LM ending with exactly the previous word

        for token in sentence.split(" "):
            stateOut = kenlm.State()
            word_score += model.BaseScore(stateIn, token, stateOut) * LOG10
            stateIn = stateOut
            current_word += token

            if token == "|":
                if current_word in known_words:
                    n_words += 1
                    ppl += word_score
                    pdf_mass_words = set(
                        compute_words_model_pdf_mass(
                            word_model, known_words_original, word_stateIn
                        )
                    )
                    if current_word not in pdf_mass_words:
                        pdf_mass_words.add(current_word)
                    ppl_lower += compute_word_logprob(
                        model, word_state, current_word
                    ) - compute_denominator(model, word_state, pdf_mass_words)
                else:
                    unk_n_words += 1

                word_stateOut = kenlm.State()
                word_model.BaseScore(
                    word_stateIn, transform_asg_back(current_word), word_stateOut
                )
                word_stateIn = word_stateOut

                current_word = ""
                word_score = 0.0
                word_state = stateOut

        stateOut = kenlm.State()
        n_words += 1
        ppl += model.BaseScore(stateIn, EOS, stateOut) * LOG10
        pdf_mass_words = set(
            compute_words_model_pdf_mass(word_model, known_words_original, word_stateIn)
        )
        if EOS not in pdf_mass_words:
            pdf_mass_words.add(EOS)
        ppl_lower += compute_word_logprob(model, word_state, EOS) - compute_denominator(
            model, word_state, pdf_mass_words
        )

        if n % 10 == 0:
            print(
                "Evaluated",
                n,
                "sentences among",
                n_logging,
                "upper limit perplexity",
                numpy.exp(-ppl / n_words),
                "lower limit perplexity",
                numpy.exp(-ppl_lower / n_words),
                "number of words",
                n_words,
                flush=True,
            )

    print("Final loss", ppl, "loss lower", ppl_lower)
    print("Upper limit on perplexity:", numpy.exp(-ppl / n_words))
    print("Lower limit on perplexity:", numpy.exp(-ppl_lower / n_words))
    print("Total number of words:", n_words, "unknown words:", unk_n_words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upper and lower limit on word perplexity for kenlm char model"
    )
    parser.add_argument(
        "--vocab_file",
        help="vocabulary of known words, use file "
        "from --limit_vocab_file during word kenLM training.",
    )
    parser.add_argument(
        "--text", help="file to evaluate, prepared for char lm training"
    )
    parser.add_argument("--char_model", help="kenlm char model")
    parser.add_argument("--word_model", help="kenlm word model")

    args = parser.parse_args()
    print("Evaluate file {}".format(args.text))

    known_words, known_words_original = prepare_vocabs(args.vocab_file)
    with open(args.text, "r") as f:
        sentences = [line.strip() for line in f]

    word_model = kenlm.LanguageModel(args.word_model)
    char_model = kenlm.LanguageModel(args.char_model)

    compute_ppl_lower_limit(
        char_model, word_model, sentences, known_words, known_words_original
    )
