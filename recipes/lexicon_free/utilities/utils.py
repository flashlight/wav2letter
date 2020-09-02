"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import re
import numpy


EOS = "</s>"


def convert_words_to_letters_asg_rep2(fin_name, fout_name):
    with open(fin_name, "r") as fin, open(fout_name, "w") as fout:
        for line in fin:
            words = line.strip().split(" ")
            for word in words:
                word = re.sub("[^a-z'.]+", "", word)
                if len(word) == 0:
                    continue
                new_word = transform_asg(word) + "|"
                fout.write(" ".join(list(new_word)) + " ")
            fout.write("\n")


def transform_asg(word):
    if word == "":
        return ""
    new_word = word[0]
    prev = word[0]
    repetition = 0
    for letter in word[1:]:
        if letter == prev:
            repetition += 1
        else:
            if repetition != 0:
                new_word += "1" if repetition == 1 else "2"
                repetition = 0
            new_word += letter
        prev = letter
    if repetition != 0:
        new_word += "1" if repetition == 1 else "2"
    return new_word


def transform_asg_back(word):
    new_word = ""
    for letter in word:
        if letter == "|":
            continue
        if letter == "1":
            new_word += new_word[-1]
        elif letter == "2":
            new_word += new_word[-1] + new_word[-1]
        else:
            new_word += letter

    return new_word


def prepare_vocabs(path):
    # read dictionary of words
    with open(path, "r") as f:
        words = f.readline().strip().split(" ")
        words = [re.sub("[^a-z'.]+", "", word) for word in words]
        known_words = set(list(map(lambda x: transform_asg(x) + "|", words))) - {""}
        words.append("</s>")
        known_words_original = set(words) - {""}
        known_words_original = numpy.array(list(known_words_original))
    return known_words, known_words_original


def prepare_vocabs_convlm(path):
    # read dictionary of words
    words = []
    with open(path, "r") as f:
        for line in f:
            word = line.strip().split(" ")[0]
            words.append(re.sub("[^a-z'.]+", "", word))
        known_words = set(list(map(lambda x: transform_asg(x) + "|", words))) - {""}
        words.append("</s>")
        known_words_original = set(words) - {""}
        known_words_original = numpy.array(list(known_words_original))
    return known_words, known_words_original
