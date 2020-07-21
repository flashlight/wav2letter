#!/usr/bin/env python3
# Perform beam-search decoding with word-level LM
# this is test with dumped acoustic model scores

import math
import os
import struct
import sys

import numpy as np
from wav2letter.common import Dictionary, create_word_dict, load_words, tkn_to_idx
from wav2letter.decoder import (
    CriterionType,
    DecoderOptions,
    KenLM,
    LexiconDecoder,
    SmearingMode,
    Trie,
)


# def ptr_as_bytes(x):
#     return struct.pack("P", x)
#
#
# def get_numpy_ptr_as_bytes(arr):
#     if not arr.flags["C_CONTIGUOUS"]:
#         raise ValueError("numpy array is not contiguous")
#     return ptr_as_bytes(arr.ctypes.data)


def read_struct(file, fmt):
    return struct.unpack(fmt, file.read(struct.calcsize(fmt)))


def load_tn(path):
    """
    Load time size and number of tokens from the dump
    (defines the score to move from token_i to token_j)

    Returns:
    --------
    int, int
    """
    with open(path, "rb") as file:
        T = read_struct(file, "i")[0]
        N = read_struct(file, "i")[0]
        return T, N


def load_emissions(path):
    """
    Load precomputed transition matrix
    (defines the score to move from token_i to token_j)

    Returns:
    --------
    numpy.array of shape [Batch=1, Time, Ntokens]
    """
    with open(path, "rb") as file:
        return np.frombuffer(file.read(T * N * 4), dtype=np.float32)


def load_transitions(path):
    """
    Load precomputed transition matrix
    (defines the score to move from token_i to token_j)

    Returns:
    --------
    numpy.array of shape [Ntokens, Ntokens]
    """
    with open(path, "rb") as file:
        return np.frombuffer(file.read(N * N * 4), dtype=np.float32)


def assert_near(x, y, tol):
    assert abs(x - y) <= tol


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} decoder_test_data_path", file=sys.stderr)
        print("  (usually: <wav2letter_root>/src/decoder/test)", file=sys.stderr)
        sys.exit(1)

    data_path = sys.argv[1]

    # load test files
    # load time and number of tokens for dumped acoustic scores
    T, N = load_tn(os.path.join(data_path, "TN.bin"))
    # load emissions [Batch=1, Time, Ntokens]
    emissions = load_emissions(os.path.join(data_path, "emission.bin"))
    # load transitions (from ASG loss optimization) [Ntokens, Ntokens]
    transitions = load_transitions(os.path.join(data_path, "transition.bin"))
    # load lexicon file, which defines spelling of words
    # the format word and its tokens spelling separated by the spaces,
    # for example for letters tokens with ASG loss:
    # ann a n 1 |
    lexicon = load_words(os.path.join(data_path, "words.lst"))
    # read lexicon and store it in the w2l dictionary
    word_dict = create_word_dict(lexicon)
    # create w2l dict with tokens set (letters in this example)
    token_dict = Dictionary(os.path.join(data_path, "letters.lst"))
    # add repetition symbol as soon as we have ASG acoustic model
    token_dict.add_entry("1")
    # create Kenlm language model
    lm = KenLM(os.path.join(data_path, "lm.arpa"), word_dict)

    # test LM
    sentence = ["the", "cat", "sat", "on", "the", "mat"]
    # start LM with nothing, get its current state
    lm_state = lm.start(False)
    total_score = 0
    lm_score_target = [-1.05971, -4.19448, -3.33383, -2.76726, -1.16237, -4.64589]
    # iterate over words in the sentence
    for i in range(len(sentence)):
        # score lm, taking current state and index of the word
        # returns new state and score for the word
        lm_state, lm_score = lm.score(lm_state, word_dict.get_index(sentence[i]))
        assert_near(lm_score, lm_score_target[i], 1e-5)
        # add score of the current word to the total sentence score
        total_score += lm_score
    # move lm to the final state, the score returned is for eos
    lm_state, lm_score = lm.finish(lm_state)
    total_score += lm_score
    assert_near(total_score, -19.5123, 1e-5)

    # build trie
    # Trie is necessary to do beam-search decoding with word-level lm
    # We restrict our search only to the words from the lexicon
    # Trie is constructed from the lexicon, each node is a token
    # path from the root to a leaf corresponds to a word spelling in the lexicon

    # get silence index
    sil_idx = token_dict.get_index("|")
    # get unknown word index
    unk_idx = word_dict.get_index("<unk>")
    # create the trie, specifying how many tokens we have and silence index
    trie = Trie(token_dict.index_size(), sil_idx)
    start_state = lm.start(False)

    # use heuristic for the trie, called smearing:
    # predict lm score for each word in the lexicon, set this score to a leaf
    # (we predict lm score for each word as each word starts a sentence)
    # word score of a leaf is propagated up to the root to have some proxy score
    # for any intermediate path in the trie
    # SmearingMode defines the function how to process scores
    # in a node came from the children nodes:
    # could be max operation or logadd or none
    for word, spellings in lexicon.items():
        usr_idx = word_dict.get_index(word)
        _, score = lm.score(start_state, usr_idx)
        for spelling in spellings:
            # max_reps should be 1; using 0 here to match DecoderTest bug
            spelling_idxs = tkn_to_idx(spelling, token_dict, 0)
            trie.insert(spelling_idxs, usr_idx, score)

    trie.smear(SmearingMode.MAX)

    # check that trie is built in consistency with c++
    trie_score_target = [-1.05971, -2.87742, -2.64553, -3.05081, -1.05971, -3.08968]
    for i in range(len(sentence)):
        word = sentence[i]
        # max_reps should be 1; using 0 here to match DecoderTest bug
        word_tensor = tkn_to_idx([c for c in word], token_dict, 0)
        node = trie.search(word_tensor)
        assert_near(node.max_score, trie_score_target[i], 1e-5)

    # Define decoder options:
    # DecoderOptions (beam_size, token_beam_size, beam_threshold, lm_weight,
    #                 word_score, unk_score, sil_score,
    #                 eos_score, log_add, criterion_type (ASG or CTC))
    opts = DecoderOptions(
        2500, 25000, 100.0, 2.0, 2.0, -math.inf, -1, 0, False, CriterionType.ASG
    )

    # define lexicon beam-search decoder with word-level lm
    # LexiconDecoder(decoder options, trie, lm, silence index,
    #                blank index (for CTC), unk index,
    #                transitiona matrix, is token-level lm)
    decoder = LexiconDecoder(opts, trie, lm, sil_idx, -1, unk_idx, transitions, False)
    # run decoding
    # decoder.decode(emissions, Time, Ntokens)
    # result is a list of sorted hypothesis, 0-index is the best hypothesis
    # each hypothesis is a struct with "score" and "words" representation
    # in the hypothesis and the "tokens" representation
    results = decoder.decode(emissions.ctypes.data, T, N)

    print(f"Decoding complete, obtained {len(results)} results")
    print("Showing top 5 results:")
    for i in range(min(5, len(results))):
        prediction = []
        for idx in results[i].tokens:
            if idx < 0:
                break
            prediction.append(token_dict.get_entry(idx))
        prediction = " ".join(prediction)
        print(f"score={results[i].score} amScore={results[i].amScore} lmScore={results[i].lmScore} prediction='{prediction}'")

    assert len(results) == 16
    hyp_score_target = [-284.0998, -284.108, -284.119, -284.127, -284.296]
    for i in range(min(5, len(results))):
        assert_near(results[i].score, hyp_score_target[i], 1e-3)
