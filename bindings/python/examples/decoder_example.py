#!/usr/bin/env python3

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
    SmearingMode,
    Trie,
    WordLMDecoder,
)


if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} decoder_test_data_path", file=sys.stderr)
    print("  (usually: <wav2letter_root>/src/decoder/test)", file=sys.stderr)
    sys.exit(1)

data_path = sys.argv[1]


def read_struct(file, fmt):
    return struct.unpack(fmt, file.read(struct.calcsize(fmt)))


def load_tn(path):
    with open(path, "rb") as file:
        T = read_struct(file, "i")[0]
        N = read_struct(file, "i")[0]
        return T, N


def load_emissions(path):
    with open(path, "rb") as file:
        return np.frombuffer(file.read(T * N * 4), dtype=np.float32)


def load_transitions(path):
    with open(path, "rb") as file:
        return np.frombuffer(file.read(N * N * 4), dtype=np.float32)


def assert_near(x, y, tol):
    assert abs(x - y) <= tol


# def ptr_as_bytes(x):
#     return struct.pack("P", x)
#
#
# def get_numpy_ptr_as_bytes(arr):
#     if not arr.flags["C_CONTIGUOUS"]:
#         raise ValueError("numpy array is not contiguous")
#     return ptr_as_bytes(arr.ctypes.data)


# load test files

T, N = load_tn(os.path.join(data_path, "TN.bin"))
emissions = load_emissions(os.path.join(data_path, "emission.bin"))
transitions = load_transitions(os.path.join(data_path, "transition.bin"))
lexicon = load_words(os.path.join(data_path, "words.lst"))
word_dict = create_word_dict(lexicon)
token_dict = Dictionary(os.path.join(data_path, "letters.lst"))
token_dict.add_entry("1")
lm = KenLM(os.path.join(data_path, "lm.arpa"), word_dict)

# test LM

sentence = ["the", "cat", "sat", "on", "the", "mat"]
lm_state = lm.start(False)
total_score = 0
lm_score_target = [-1.05971, -4.19448, -3.33383, -2.76726, -1.16237, -4.64589]
for i in range(len(sentence)):
    lm_state, lm_score = lm.score(lm_state, word_dict.get_index(sentence[i]))
    assert_near(lm_score, lm_score_target[i], 1e-5)
    total_score += lm_score
lm_state, lm_score = lm.finish(lm_state)
total_score += lm_score
assert_near(total_score, -19.5123, 1e-5)

# build trie

sil_idx = token_dict.get_index("|")
unk_idx = word_dict.get_index("<unk>")
trie = Trie(token_dict.index_size(), sil_idx)
start_state = lm.start(False)

for word, spellings in lexicon.items():
    usr_idx = word_dict.get_index(word)
    _, score = lm.score(start_state, usr_idx)
    for spelling in spellings:
        # max_reps should be 1; using 0 here to match DecoderTest bug
        spelling_idxs = tkn_to_idx(spelling, token_dict, 0)
        trie.insert(spelling_idxs, usr_idx, score)

trie.smear(SmearingMode.MAX)

trie_score_target = [-1.05971, -2.87742, -2.64553, -3.05081, -1.05971, -3.08968]
for i in range(len(sentence)):
    word = sentence[i]
    # max_reps should be 1; using 0 here to match DecoderTest bug
    word_tensor = tkn_to_idx([c for c in word], token_dict, 0)
    node = trie.search(word_tensor)
    assert_near(node.max_score, trie_score_target[i], 1e-5)

opts = DecoderOptions(2500, 100.0, 2.0, 2.0, -math.inf, False, -1, CriterionType.ASG)

decoder = WordLMDecoder(opts, trie, lm, sil_idx, -1, unk_idx, transitions)
results = decoder.decode(emissions.ctypes.data, T, N)

print(f"Decoding complete, obtained {len(results)} results")
print("Showing top 5 results:")
for i in range(5):
    prediction = []
    for idx in results[i].tokens:
        if idx < 0:
            break
        prediction.append(token_dict.get_entry(idx))
    prediction = " ".join(prediction)
    print(f"score={results[i].score} prediction='{prediction}'")

assert len(results) == 1452
hyp_score_target = [-278.111, -278.652, -279.275, -279.847, -280.01]
for i in range(5):
    assert_near(results[i].score, hyp_score_target[i], 1e-3)
