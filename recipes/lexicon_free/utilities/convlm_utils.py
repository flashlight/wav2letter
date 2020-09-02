"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from fairseq import options
from fairseq.models.fconv import FConvDecoder


EOS = '</s>'
UNK = '<unk>'
EOSIDX = 2
UNKIDX = 3


def compute_new_state(model_state):
    new_state = dict()
    for key, val in model_state["model"].items():
        if "1.weight" in key and "adaptive" in key:
            new_state[
                ".".join(key.split(".")[1:]).replace("1.weight", "2.weight")
            ] = val
        else:
            new_state[".".join(key.split(".")[1:])] = val
    return new_state


def load_char_model_20B(pytorch_model_path, fairseq_dict, dataset_type):
    layer = eval(
        "[(512, 5)] + [(128, 1, 0), (128, 5, 0), (256, 1, 3)] * 3 + "
        "[(256, 1, 0), (256, 5, 0), (512, 1, 3)] * 3 + "
        "[(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3 + "
        "[(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 9 + "
        "[(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]"
    )
    model_state = torch.load(pytorch_model_path)

    convLM_char = FConvDecoder(
        fairseq_dict,
        embed_dim=256,
        out_embed_dim=256,
        max_positions=1024,
        convolutions=layer,
        dropout=0.1,
        share_embed=False,
        attention=False,
        positional_embeddings=False,
        adaptive_softmax_cutoff=None,
        adaptive_softmax_dropout=0,
    ).cuda()

    convLM_char.load_state_dict(compute_new_state(model_state))
    convLM_char.eval()
    return convLM_char


def load_char_model_14B(pytorch_model_path, fairseq_dict, dataset_type):
    layer = eval(
        "[(512, 5)] + [(128, 1, 0), (128, 5, 0), (512, 1, 3)] * 3 + "
        "[(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3 + "
        "[(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 6 + "
        "[(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]"
    )
    model_state = torch.load(pytorch_model_path)

    convLM_char = FConvDecoder(
        fairseq_dict,
        embed_dim=128,
        out_embed_dim=128,
        max_positions=1024,
        convolutions=layer,
        dropout=0.1,
        share_embed=False,
        attention=False,
        positional_embeddings=False,
        adaptive_softmax_cutoff=None,
        adaptive_softmax_dropout=0,
    ).cuda()

    convLM_char.load_state_dict(compute_new_state(model_state))
    convLM_char.eval()
    return convLM_char


def load_word_model(pytorch_model_path, fairseq_dict, dataset_type):
    layer = eval(
        "[(512, 5)] + [(128, 1, 0), (128, 5, 0), (512, 1, 3)] * 3 + "
        "[(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3 + "
        "[(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 6 + "
        "[(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]"
    )
    model_state = torch.load(pytorch_model_path)

    if dataset_type == "wsj":
        cutoff = "10000,50000,100000"
    elif dataset_type == "ls":
        cutoff = "10000,50000,200000"
    else:
        cutoff = ""
    convLM = FConvDecoder(
        fairseq_dict,
        embed_dim=128,
        out_embed_dim=128,
        max_positions=1024,
        convolutions=layer,
        dropout=0.1,
        share_embed=False,
        attention=False,
        positional_embeddings=False,
        adaptive_softmax_cutoff=(options.eval_str_list(cutoff, type=int)),
        adaptive_softmax_dropout=0,
    ).cuda()

    convLM.load_state_dict(compute_new_state(model_state))
    convLM.eval()
    convLM.adaptive_softmax.eval()
    return convLM


def decodeInputText(sentences, token_indices_dict):
    sentences_decoded = []
    for line in sentences:
        sentences_decoded.append(
            [
                token_indices_dict[UNK]
                if token not in token_indices_dict
                else token_indices_dict[token]
                for token in line.split(" ")
            ]
        )
    return sentences_decoded


def build_token_index_correspondence(dict_fname):
    # follow fairseq
    token_indices_dict = dict()
    indices_token_dict = dict()
    with open(dict_fname, "r") as f:
        for index, line in enumerate(f):
            token_indices_dict[line.strip().split(" ")[0]] = index + 4
            indices_token_dict[index + 4] = line.strip().split(" ")[0]
    token_indices_dict[EOS] = EOSIDX
    indices_token_dict[EOSIDX] = EOS
    token_indices_dict[UNK] = UNKIDX
    indices_token_dict[UNKIDX] = UNK
    return token_indices_dict, indices_token_dict
