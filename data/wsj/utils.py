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


def preprocess_word(word):
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
    word = word.replace("*", "") if re.match(r"^\*\S+\*", word) else word
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

    return word


def find_transcripts(dst_paths):
    transcripts = dict()
    for ds_path in dst_paths:
        for dirpath, _, filenames in os.walk(ds_path):
            for filename in filenames:
                if not filename.endswith(".dot"):
                    continue
                full_path = os.path.join(dirpath, filename)
                subset = full_path.split(os.sep)[-3]
                assert subset, "Subset is empty"

                transcripts.setdefault(subset, dict())
                with open(full_path, "r") as f:
                    for line in f:
                        transcript, file_id = line.strip().rsplit(" ", 1)
                        file_id = file_id.strip("()")
                        if not transcript or not file_id:
                            continue

                        if subset in transcripts and file_id in transcripts[subset]:
                            assert (
                                transcripts[subset][file_id] == transcript
                            ), "different transcriptions available for {i}".format(
                                i=file_id
                            )
                        transcripts[subset][file_id] = transcript
    return transcripts


def ndx_to_samples(prefix, filename, transcripts, transform=None, sep="-"):
    samples_list = []
    with open(os.path.join(prefix, filename), "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if transform is not None:
                line = transform(line)
            if line is None:
                continue
            pre, suf = line.split(":")
            p1, p2, p3 = pre.split("_")
            suf = suf.lstrip(" /")
            ds, subset, _, sample_id = suf.replace(".wv1", "").rsplit("/", 3)

            fname = os.path.join(prefix, "{}{}{}.{}".format(p1, sep, p2, p3), suf)

            assert os.path.exists(fname), "Audio file {} doesn't exist".format(fname)
            assert (
                subset in transcripts
            ), "Subset {} is absent in the transcription".format(subset)
            assert (
                sample_id in transcripts[subset]
            ), "Id {} is absent in the subset {} of transcription for file {}".format(
                sample_id, subset, fname
            )

            samples_list.append(
                {
                    "id": sample_id,
                    "filename": fname,
                    "subset": subset,
                    "transcript": transcripts[subset][sample_id],
                    "basename": os.path.join("{}{}{}.{}".format(p1, sep, p2, p3), suf),
                }
            )
    samples_list.sort(key=lambda x: x["id"])
    return samples_list


def convert_to_flac(sample_data):
    sample, idx, dst, sph2pipe = sample_data
    filename = sample["filename"]
    out_prefix = os.path.join(dst, "%09d" % idx)

    # flac
    if not os.path.exists(out_prefix + ".flac"):
        tmp_file = os.path.join(dst, "{pid}_tmp.wav".format(pid=os.getpid()))
        os.system("{sph} -f wav {i} {o}".format(sph=sph2pipe, i=filename, o=tmp_file))
        assert (
            sox.file_info.duration(tmp_file) > 0
        ), "Audio file {} duration is zero.".format(filename)

        sox_tfm = sox.Transformer()
        sox_tfm.set_output_format(file_type="flac", encoding="signed-integer", bits=16)
        sox_tfm.build(tmp_file, out_prefix + ".flac")

        os.remove(tmp_file)

    duration = sox.file_info.duration(out_prefix + ".flac") * 1000  # miliseconds
    transcript = " ".join(
        [preprocess_word(word) for word in sample["transcript"].split()]
    )
    transcript = re.sub(" +", " ", transcript).strip()

    return [sample["basename"], out_prefix + ".flac", str(duration), transcript]
