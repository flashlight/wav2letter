"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re

import sox


def transcript2wordspelling(transcript, filename):
    words = []
    spellings = []
    for token in transcript.split():
        word, spelling = preprocess(token)
        if word and spelling:
            assert re.match(
                r"[a-z']+", spelling
            ), "invalid transcript '{t}' for file '{f}'".format(
                t=transcript, f=filename
            )
            words.append(word)
            spellings.append(spelling)
    return " ".join(words), " | ".join([" ".join(s) for s in spellings])


def find_transcripts(ds_path, transcripts):
    for dirpath, _, filenames in os.walk(ds_path):
        for filename in filenames:
            if not filename.endswith(".dot"):
                continue
            full_path = os.path.join(dirpath, filename)
            subset = full_path.split(os.sep)[-3]
            assert subset

            transcripts.setdefault(subset, {})
            with open(full_path, "r") as f:
                for line in f:
                    transcript, id = line.strip().rsplit(" ", 1)
                    id = id.strip("()")

                    if not transcript or not id:
                        continue

                    if subset in transcripts and id in transcripts[subset]:
                        assert (
                            transcripts[subset][id] == transcript
                        ), "different transcriptions available for {i}".format(i=id)
                    transcripts[subset][id] = transcript
    return transcripts


def ndx2idlist(prefix, filename, transcripts, transform=None, list=None):
    list = list or []
    with open(os.path.join(prefix, filename), "r") as f:
        for line in f:
            line = line.strip()
            if transform is not None:
                line = transform(line)

            if not line or line.startswith(";"):
                continue

            pre, suf = line.split(":")
            p1, p2, p3 = pre.split("_")
            suf = suf.lstrip(" /")
            ds, subset, _, id = suf.replace(".wv1", "").rsplit("/", 3)

            sep = "-" if ds == "wsj0" else "_"
            fname = os.path.join(prefix, "{}{}{}.{}".format(p1, sep, p2, p3), suf)

            assert os.path.exists(fname)
            assert subset in transcripts and id in transcripts[subset]

            list.append(
                {
                    "id": id,
                    "filename": fname,
                    "subset": subset,
                    "transcript": transcripts[subset][id],
                }
            )
    list.sort(key=lambda x: x["id"])
    return list


def preprocess(word):
    word = re.sub(r"^~$", "", word)
    word = re.sub(r"^~~$", "", word)
    word = re.sub(r"\\", "", word)
    word = re.sub(r"^\[<\S+\]$", "", word)
    word = re.sub(r"^\[\S+>\]$", "", word)
    word = re.sub(r"^\[\S+/\]$", "", word)
    word = re.sub(r"^\[/\S+\]$", "", word)
    word = re.sub(r"^\[\S+\]$", "", word)  # NOISE
    if re.match(r"^<\S+>$", word) and word != "<NOISE>":
        word = word[1:-1]
    word = word.replace("*", "") if re.match("^\*\S+\*", word) else word
    word = re.sub(r"^%PERCENT$", "PERCENT", word)
    word = re.sub(r"^\.POINT$", "POINT", word)
    word = re.sub(r"`", "'", word)  # typo
    word = re.sub(r"^\(IN\-PARENTHESIS$", "(IN-PARENTHESES", word)  # mispell
    word = re.sub(r"^Corp;$", "Corp", word)  # mispell
    word = re.sub(r"^\-\-DASH$", "-DASH", word)  # mispell
    if word != ":COLON":
        word = word.replace(":", "")  # some emphasis stuff
    if word != "!EXCLAMATION-POINT":
        word = word.replace("!", "")  # some emphasis stuff
    word = re.sub(r"^\.$", "", word)
    word = word.lower()

    if not word:
        return "", ""

    spelling = re.sub(r"\(\S+\)", "", word)  # not pronounced
    spelling = re.sub('[,\.:\-/&\?\!\(\)";\{\}\_#]+', "", spelling)

    if word == "'single-quote":
        spelling = spelling.replace("'", "")

    return word, spelling


def write_sample(sample_data):
    sample, idx, dst, sph2pipe = sample_data
    filename = sample["filename"]
    transcript = sample["transcript"]
    words, spellings = transcript2wordspelling(transcript, filename)

    out_prefix = os.path.join(dst, "%09d" % idx)

    # flac
    tmp_file = os.path.join(dst, "{pid}_tmp.wav".format(pid=os.getpid()))
    os.system("{sph} -f wav {i} {o}".format(sph=sph2pipe, i=filename, o=tmp_file))
    assert sox.file_info.duration(tmp_file) > 0

    sox_tfm = sox.Transformer()
    sox_tfm.set_output_format(file_type="flac", encoding="signed-integer", bits=16)
    sox_tfm.build(tmp_file, out_prefix + ".flac")

    os.remove(tmp_file)

    # words
    with open(out_prefix + ".wrd", "w") as f:
        f.write(words)

    # ltr
    with open(out_prefix + ".tkn", "w") as f:
        f.write(spellings)

    # id
    with open(out_prefix + ".id", "w") as f:
        f.write("file_id\t{idx}".format(idx=idx))


def processdict(filename):
    d = {}
    with open(filename, "r") as f:
        for line in f:
            if not line.startswith(";;;"):
                continue
            word = line.split()[0]
            d[word] = True
    return d
