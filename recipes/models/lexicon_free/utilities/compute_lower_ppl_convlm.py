"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Compute upper and lower limits on word perplexity for convlm models

Command : python3 compute_lower_ppl_convlm.py --model [...] --dict [...] \
            --word_model [...] --word_dict [...] \
            --text [...] --model_type char14B --dataset_type ls

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import numpy
import torch
from convlm_utils import (
    EOS,
    EOSIDX,
    UNKIDX,
    build_token_index_correspondence,
    decodeInputText,
    load_char_model_14B,
    load_char_model_20B,
    load_word_model,
)
from fairseq.data import Dictionary
from utils import prepare_vocabs_convlm, transform_asg, transform_asg_back


# reusing previous states for some reason is slower than reevaluating the full sentence.
# TODO speedup with batching and using previous state
def compute_word_logprob(model, current_state, target_word, token_index_dict):
    if target_word == EOS:
        x = torch.LongTensor(current_state).reshape(1, len(current_state)).cuda()
        with torch.no_grad():
            y = model.forward(x)[0]
            logprobs = torch.nn.functional.log_softmax(y, 2).detach().cpu().numpy()[0]
        return logprobs[-1, EOSIDX]
    else:
        additional_state = [token_index_dict[token] for token in list(target_word)]
        with torch.no_grad():
            x = (
                torch.LongTensor(current_state + additional_state[:-1])
                .reshape(1, len(current_state) + len(additional_state) - 1)
                .cuda()
            )
            y = model.forward(x)[0]
            logprobs = torch.nn.functional.log_softmax(y, 2).detach().cpu().numpy()[0]
        return numpy.sum(
            logprobs[-len(additional_state) :][
                numpy.arange(len(additional_state)), additional_state
            ]
        )


def compute_denominator(model, current_state, words, token_index_dict):
    preds = [
        compute_word_logprob(model, current_state, word, token_index_dict)
        for word in words
    ]
    assert len(preds) != 0, "Invalid denominator"
    max_pred = numpy.max(preds)
    return max_pred + numpy.log(numpy.sum(numpy.exp(preds - max_pred)))


def compute_words_model_pdf_mass(
    word_probs, current_state_position, known_words, known_words_decoded
):
    probs = word_probs[current_state_position, known_words_decoded]
    indices = numpy.argsort(-probs)
    # unk word is not added to this pdf mass, sometimes its prob is huge
    # so take percentile from known word pdf
    probs_sum = numpy.sum(probs)
    top = numpy.where(numpy.cumsum(probs[indices]) > 0.95 * probs_sum)[0][0]
    return [
        transform_asg(w) + "|" if w != EOS else w for w in known_words[indices[:top]]
    ]


def compute_ppl_lower_limit(
    model,
    word_model,
    sentences,
    known_words,
    known_words_original,
    known_words_original_decoded,
    indices_token_dict,
    token_indices_dict,
):
    n_words = 0
    unk_n_words = 0
    ppl = 0.0
    ppl_lower = 0.0

    n_logging = len(sentences)
    for n, sentence in enumerate(sentences):
        current_state = [EOSIDX]
        current_word = ""
        current_word_state_position = 0
        addition_state = []

        wordLM_sentence = (
            "".join([indices_token_dict[idx] for idx in sentence])
            .replace("|", " ")
            .strip()
        )
        wordLM_sentence = [
            transform_asg_back(word) for word in wordLM_sentence.split(" ")
        ]
        wordLM_sentence_decoded = [EOSIDX] + [
            UNKIDX if word not in word_indices_dict else word_indices_dict[word]
            for word in wordLM_sentence
        ]
        with torch.no_grad():
            x = (
                torch.LongTensor(wordLM_sentence_decoded)
                .reshape(1, len(wordLM_sentence_decoded))
                .cuda()
            )
            y = word_model.forward(x)[0]
            words_probs = numpy.exp(
                word_model.adaptive_softmax.get_log_prob(y, None)
                .detach()
                .cpu()
                .numpy()[0]
            )

        for token_idx in sentence:
            current_word += indices_token_dict[token_idx]
            addition_state.append(token_idx)

            if indices_token_dict[token_idx] == "|":
                if current_word in known_words:
                    n_words += 1
                    pdf_mass_words = set(
                        compute_words_model_pdf_mass(
                            words_probs,
                            current_word_state_position,
                            known_words_original,
                            known_words_original_decoded,
                        )
                    )
                    if current_word not in pdf_mass_words:
                        pdf_mass_words.add(current_word)

                    word_score = compute_word_logprob(
                        model, current_state, current_word, token_indices_dict
                    )
                    ppl += word_score
                    ppl_lower += word_score - compute_denominator(
                        model, current_state, pdf_mass_words, token_indices_dict
                    )
                else:
                    unk_n_words += 1

                current_word = ""
                current_state += addition_state
                addition_state = []
                current_word_state_position += 1

        # process eos
        word_score = compute_word_logprob(model, current_state, EOS, token_indices_dict)
        n_words += 1
        ppl += word_score
        pdf_mass_words = set(
            compute_words_model_pdf_mass(
                words_probs,
                current_word_state_position,
                known_words_original,
                known_words_original_decoded,
            )
        )

        if EOS not in pdf_mass_words:
            pdf_mass_words.add(EOS)
        ppl_lower += word_score - compute_denominator(
            model, current_state, pdf_mass_words, token_indices_dict
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
        description="Upper and lower limits on word perplexity for convlm models"
    )
    parser.add_argument("--model", help="path to convlm model")
    parser.add_argument("--word_model", help="path to convlm model")
    parser.add_argument("--dict", help="path to convlm dict file in data")
    parser.add_argument(
        "--word_dict", help="path to convlm word convlm dict file in data"
    )
    parser.add_argument(
        "--text", help="file to evaluate, in necessary format for model"
    )
    parser.add_argument("--model_type", help='"char14B" or "char20B"')
    parser.add_argument("--dataset_type", help='"ls" or "wsj"', default="ls")

    args = parser.parse_args()
    print("Evaluate file {}".format(args.text))

    token_indices_dict, indices_token_dict = build_token_index_correspondence(args.dict)
    word_indices_dict, indices_word_dict = build_token_index_correspondence(
        args.word_dict
    )

    known_words, known_words_original = prepare_vocabs_convlm(args.word_dict)
    known_words_original_decoded = numpy.array(
        [
            UNKIDX if w not in word_indices_dict else word_indices_dict[w]
            for w in known_words_original
        ]
    )
    with open(args.text, "r") as f:
        sentences = [line.strip() for line in f]

    input_data = decodeInputText(sentences, token_indices_dict)
    fairseq_dict = Dictionary.load(args.dict)
    word_fairseq_dict = Dictionary.load(args.word_dict)

    word_model = load_word_model(args.word_model, word_fairseq_dict, args.dataset_type)
    if "14B" in args.model_type:
        char_model = load_char_model_14B(args.model, fairseq_dict, args.dataset_type)
    else:
        char_model = load_char_model_20B(args.model, fairseq_dict, args.dataset_type)
    compute_ppl_lower_limit(
        char_model,
        word_model,
        input_data,
        known_words,
        known_words_original,
        known_words_original_decoded,
        indices_token_dict,
        token_indices_dict,
    )
