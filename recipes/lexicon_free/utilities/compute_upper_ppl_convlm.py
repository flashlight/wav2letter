"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Compute upper limit on word perplexity for convlm models

Command (for word) : python3 compute_upper_ppl_convlm.py --model [...] \
                        --dict [...] --text [...] --model_type word --dataset_type ls
Command (for char) : python3 compute_upper_ppl_convlm.py --model [...] \
                        --dict [...] --word_dict [...] --text [...] \
                        --model_type char14B --dataset_type ls
Command (for char) : python3 compute_upper_ppl_convlm.py --model [...] \
                        --dict [...] --word_dict [...] --text [...] \
                        --model_type char20B --dataset_type ls

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import numpy
import torch
from convlm_utils import (
    EOSIDX,
    UNKIDX,
    build_token_index_correspondence,
    decodeInputText,
    load_char_model_14B,
    load_char_model_20B,
    load_word_model,
)
from fairseq.data import Dictionary
from utils import transform_asg


def compute_ppl_upper_limit_char_convlm(
    model,
    input_charlm,
    charLM_indices_token_dict,
    charLM_token_indices_dict,
    known_words,
):
    sum_logp = 0
    n_words = 0
    sum_logp_known = 0
    n_known_words = 0
    sum_logp_unknown = 0
    n_unknown_words = 0
    n_letters = 0

    for sentence in input_charlm:
        x = torch.LongTensor([EOSIDX] + sentence).reshape(1, len(sentence) + 1).cuda()
        with torch.no_grad():
            y = model.forward(x)[0]
            logprobs = torch.nn.functional.log_softmax(y, 2).detach().cpu().numpy()[0]

        current_word = ""
        word_ppl = 0.0
        for index, token_id in enumerate(sentence):
            n_letters += 1
            current_word += charLM_indices_token_dict[token_id]
            word_ppl += logprobs[index, token_id]
            if charLM_indices_token_dict[token_id] == "|":
                if current_word in known_words:
                    sum_logp_known += word_ppl
                    n_known_words += 1
                else:
                    sum_logp_unknown += word_ppl
                    n_unknown_words += 1
                current_word = ""
                word_ppl = 0

        sum_logp += numpy.sum(logprobs[numpy.arange(len(sentence)), sentence])
        n_words += numpy.sum(numpy.array(sentence) == charLM_token_indices_dict["|"])
        # add eos
        sum_logp += logprobs[-1, EOSIDX]
        n_words += 1
        sum_logp_known += logprobs[-1, EOSIDX]
        n_known_words += 1
        n_letters += 1

    loss_letter = -(sum_logp + sum_logp_unknown) / n_letters
    ppl_word_no_unk = numpy.exp(-sum_logp_known / n_known_words)
    ppl_word_unk = numpy.exp(-sum_logp_unknown / n_unknown_words)
    assert n_known_words + n_unknown_words == n_words, "Error in words counting"
    assert numpy.allclose(sum_logp, sum_logp_known + sum_logp_unknown), "Error in loss"

    ppl_word = numpy.exp(-sum_logp / n_words)

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


def compute_ppl_upper_limit_word_convlm(model, input_wordlm):
    sum_logp_known = 0
    n_known_words = 0
    sum_logp_unknown = 0
    n_unknown_words = 0

    for sentence in input_wordlm:
        x = torch.LongTensor([EOSIDX] + sentence).reshape(1, len(sentence) + 1).cuda()
        with torch.no_grad():
            y = model.forward(x)[0]
            logprobs = (
                model.adaptive_softmax.get_log_prob(y, None).detach().cpu().numpy()[0]
            )

        for index, token_id in enumerate(sentence):
            if token_id != UNKIDX:
                sum_logp_known += logprobs[index, token_id]
                n_known_words += 1
            else:
                sum_logp_unknown += logprobs[index, token_id]
                n_unknown_words += 1

        # add eos
        sum_logp_known += logprobs[-1, EOSIDX]
        n_known_words += 1

    ppl_word_no_unk = numpy.exp(-sum_logp_known / n_known_words)
    ppl_word_unk = numpy.exp(-sum_logp_unknown / n_unknown_words)
    ppl_word = numpy.exp(
        -(sum_logp_known + sum_logp_unknown) / (n_known_words + n_unknown_words)
    )

    print("Word perplexity for all words: {}".format(ppl_word))
    print("Word perplexity for unknown words: {}".format(ppl_word_unk))
    print(
        "(Reported in the paper) "
        "Word perplexity for known words: {}".format(ppl_word_no_unk)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upper limit on word perplexity for convlm models"
    )
    parser.add_argument("--model", help="path to convlm model")
    parser.add_argument("--dict", help="path to convlm dict file in data")
    parser.add_argument(
        "--text", help="file to evaluate, in necessary format for model"
    )
    parser.add_argument("--model_type", help='"word" or "char14B" or "char20B"')
    parser.add_argument("--dataset_type", help='"ls" or "wsj"', default="ls")
    parser.add_argument(
        "--word_dict",
        help="path to convlm word convlm dict file"
        "in data (ignored for word model eval)",
        default=None,
    )

    args = parser.parse_args()
    print("Evaluate file {}".format(args.text))

    token_indices_dict, indices_token_dict = build_token_index_correspondence(args.dict)

    with open(args.text, "r") as f:
        sentences = [line.strip() for line in f]

    input_data = decodeInputText(sentences, token_indices_dict)
    fairseq_dict = Dictionary.load(args.dict)

    if args.model_type == "word":
        model = load_word_model(args.model, fairseq_dict, args.dataset_type)
        compute_ppl_upper_limit_word_convlm(model, input_data)
    else:
        with open(args.word_dict, "r") as f:
            known_words = set(
                [transform_asg(line.strip().split(" ")[0]) + "|" for line in f]
            )
        if "14B" in args.model_type:
            model = load_char_model_14B(args.model, fairseq_dict, args.dataset_type)
        else:
            model = load_char_model_20B(args.model, fairseq_dict, args.dataset_type)
        compute_ppl_upper_limit_char_convlm(
            model, input_data, indices_token_dict, token_indices_dict, known_words
        )
