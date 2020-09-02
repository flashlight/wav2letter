"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to package original WSJ datasets into a form readable in wav2letter++
pipelines

Please install `sph2pipe` on your own -
see https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools \
  with commands :

  wget https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz
  tar -xzf sph2pipe_v2.5.tar.gz && cd sph2pipe_v2.5
  gcc -o sph2pipe *.c -lm

Command : python3 prepare.py --wsj0 [...]/WSJ0/media \
    --wsj1 [...]/WSJ1/media --dst [...] --sph2pipe [...]/sph2pipe_v2.5/sph2pipe

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re
import subprocess
from multiprocessing import Pool

import numpy
from tqdm import tqdm
from utils import convert_to_flac, find_transcripts, ndx_to_samples, preprocess_word


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSJ Dataset creation.")
    parser.add_argument("--wsj0", help="top level directory containing all WSJ0 discs")
    parser.add_argument("--wsj1", help="top level directory containing all WSJ1 discs")
    parser.add_argument("--dst", help="destination directory", default="./wsj")
    parser.add_argument(
        "--wsj1_type",
        help="if you are using larger corpus LDC94S13A, set parameter to `LDC94S13A`",
        default="LDC94S13B",
    )

    parser.add_argument(
        "--sph2pipe",
        help="path to sph2pipe executable",
        default="./sph2pipe_v2.5/sph2pipe",
    )
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )

    args = parser.parse_args()
    wsj1_sep = "-" if args.wsj1_type == "LDC94S13A" else "_"

    assert os.path.isdir(str(args.wsj0)), "WSJ0 directory is not found - '{d}'".format(
        d=args.wsj0
    )
    assert os.path.isdir(str(args.wsj1)), "WSJ1 directory is not found - '{d}'".format(
        d=args.wsj1
    )
    assert args.wsj0 != args.wsj1, "WSJ0 and WSJ1 directories can't be the same"
    assert os.path.exists(args.sph2pipe), "sph2pipe not found '{d}'".format(
        d=args.sph2pipe
    )

    # Prepare audio data
    transcripts = find_transcripts([args.wsj0, args.wsj1])

    subsets = dict()
    subsets["si84"] = ndx_to_samples(
        args.wsj0,
        "11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx",
        transcripts,
        lambda line: None if "11_2_1:wsj0/si_tr_s/401" in line else line,
    )
    assert len(subsets["si84"]) == 7138, "Incorrect number of samples in si84 part:"
    " should be 7138, but fould #{}.".format(len(subsets["si84"]))

    subsets["si284"] = ndx_to_samples(
        args.wsj0,
        "11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx",
        transcripts,
        lambda line: None if "11_2_1:wsj0/si_tr_s/401" in line else line,
    )

    subsets["si284"] = subsets["si284"] + ndx_to_samples(
        args.wsj1,
        "13{}34.1/wsj1/doc/indices/si_tr_s.ndx".format(wsj1_sep),
        transcripts,
        None,
        wsj1_sep,
    )
    assert len(subsets["si284"]) == 37416, "Incorrect number of samples in si284 part: "
    "should be 37416, but fould {}.".format(len(subsets["si284"]))

    subsets["nov92"] = ndx_to_samples(
        args.wsj0,
        "11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx",
        transcripts,
        lambda line: line + ".wv1",
    )
    assert (
        len(subsets["nov92"]) == 333
    ), "Incorrect number of samples in si284 part: should be 333, but fould {}.".format(
        len(subsets["nov92"])
    )

    subsets["nov92_5k"] = ndx_to_samples(
        args.wsj0,
        "11-13.1/wsj0/doc/indices/test/nvp/si_et_05.ndx",
        transcripts,
        lambda line: line + ".wv1",
    )
    assert (
        len(subsets["nov92_5k"]) == 330
    ), "Incorrect number of samples in si284 part: should be 330, but fould {}.".format(
        len(subsets["nov92_5k"])
    )

    subsets["nov93"] = ndx_to_samples(
        args.wsj1,
        "13{}32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx".format(wsj1_sep),
        transcripts,
        lambda line: line.replace("13_32_1", "13_33_1"),
        wsj1_sep,
    )
    assert (
        len(subsets["nov93"]) == 213
    ), "Incorrect number of samples in si284 part: should be 213, but fould {}.".format(
        len(subsets["nov93"])
    )

    subsets["nov93_5k"] = ndx_to_samples(
        args.wsj1,
        "13{}32.1/wsj1/doc/indices/wsj1/eval/h2_p0.ndx".format(wsj1_sep),
        transcripts,
        lambda line: line.replace("13_32_1", "13_33_1"),
        wsj1_sep,
    )
    assert (
        len(subsets["nov93_5k"]) == 215
    ), "Incorrect number of samples in si284 part: should be 215, but fould {}.".format(
        len(subsets["nov93_5k"])
    )

    subsets["nov93dev"] = ndx_to_samples(
        args.wsj1,
        "13{}34.1/wsj1/doc/indices/h1_p0.ndx".format(wsj1_sep),
        transcripts,
        None,
        wsj1_sep,
    )
    assert (
        len(subsets["nov93dev"]) == 503
    ), "Incorrect number of samples in si284 part: should be 503, but fould {}.".format(
        len(subsets["nov93dev"])
    )

    subsets["nov93dev_5k"] = ndx_to_samples(
        args.wsj1,
        "13{}34.1/wsj1/doc/indices/h2_p0.ndx".format(wsj1_sep),
        transcripts,
        None,
        wsj1_sep,
    )
    assert (
        len(subsets["nov93dev_5k"]) == 513
    ), "Incorrect number of samples in si284 part: should be 513, but fould {}.".format(
        len(subsets["nov93dev_5k"])
    )

    audio_path = os.path.join(args.dst, "audio")
    text_path = os.path.join(args.dst, "text")
    lists_path = os.path.join(args.dst, "lists")
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(lists_path, exist_ok=True)
    transcription_words = set()

    for set_name, samples in subsets.items():
        n_samples = len(samples)
        print(
            "Writing {s} with {n} samples\n".format(s=set_name, n=n_samples), flush=True
        )
        data_dst = os.path.join(audio_path, set_name)
        if os.path.exists(data_dst):
            print(
                """The folder {} exists, existing flac for this folder will be skipped for generation.
                Please remove the folder if you want to regenerate the data""".format(
                    data_dst
                ),
                flush=True,
            )
        with Pool(args.process) as p:
            os.makedirs(data_dst, exist_ok=True)
            samples_info = list(
                tqdm(
                    p.imap(
                        convert_to_flac,
                        zip(
                            samples,
                            numpy.arange(n_samples),
                            [data_dst] * n_samples,
                            [args.sph2pipe] * n_samples,
                        ),
                    ),
                    total=n_samples,
                )
            )
            list_dst = os.path.join(lists_path, set_name + ".lst")
            if not os.path.exists(list_dst):
                with open(list_dst, "w") as f_list:
                    for sample_info in samples_info:
                        f_list.write(" ".join(sample_info) + "\n")
            else:
                print(
                    "List {} already exists, skip its generation."
                    " Please remove it if you want to regenerate the list".format(
                        list_dst
                    ),
                    flush=True,
                )

        for sample_info in samples_info:
            transcription_words.update(sample_info[3].lower().split(" "))
        # Prepare text data
        text_dst = os.path.join(text_path, set_name + ".txt")
        if not os.path.exists(text_dst):
            with open(text_dst, "w") as f_text:
                for sample_info in samples_info:
                    f_text.write(sample_info[3] + "\n")
        else:
            print(
                "Transcript text file {} already exists, skip its generation."
                " Please remove it if you want to regenerate the list".format(text_dst),
                flush=True,
            )

    # Prepare text data (for language model)
    lm_paths = [
        "13{}32.1/wsj1/doc/lng_modl/lm_train/np_data/87".format(wsj1_sep),
        "13{}32.1/wsj1/doc/lng_modl/lm_train/np_data/88".format(wsj1_sep),
        "13{}32.1/wsj1/doc/lng_modl/lm_train/np_data/89".format(wsj1_sep),
    ]
    if not os.path.exists(os.path.join(text_path, "cmudict.0.7a")):
        url = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict.0.7a"
        cmd = "cd {} && wget {}".format(text_path, url)
        os.system(cmd)
    else:
        print("CMU dict already exists, skip its downloading", flush=True)

    allowed_words = []
    with open(os.path.join(text_path, "cmudict.0.7a"), "r") as f_cmu:
        for line in f_cmu:
            line = line.strip()
            if line.startswith(";;;"):
                continue
            allowed_words.append(line.split(" ")[0].lower())

    lm_file = os.path.join(text_path, "lm.txt")
    # define valid words for correct splitting into sentences with "."
    existed_words = set.union(set(allowed_words), transcription_words)
    existed_words = existed_words - {"prof."}  # for reproducibility from lua code

    if os.path.exists(lm_file):
        print(
            "LM data already exist, skip its generation."
            " Please remove the file {} to regenerate it".format(lm_file),
            flush=True,
        )
    else:
        with open(lm_file, "w") as f_lm:
            for path in lm_paths:
                path = os.path.join(args.wsj1, path)
                for filename in os.listdir(path):
                    if not filename.endswith(".z"):
                        continue
                    # Get text from zip files
                    filename = os.path.join(path, filename)
                    process = subprocess.Popen(
                        ["zcat", filename], stdout=subprocess.PIPE
                    )
                    out, _ = process.communicate()
                    assert process.returncode == 0, "Error during zcat"
                    text_data = out.decode("utf-8")
                    text_data = text_data.lower()
                    # split several sentences into sequence (split if word contains
                    # dot only at the end and this word is absent
                    # in the existed words set)
                    text_data = " ".join(
                        [
                            word[:-1] + "\n"
                            if len(word) > 2
                            and word[-1] == "."
                            and "." not in word[:-1]
                            and word not in existed_words
                            else word
                            for word in text_data.split()
                        ]
                    )

                    text_data = re.sub("<s[^>]+>", "<s>", text_data)
                    text_data = re.sub("<s>", "{", text_data)
                    text_data = re.sub("</s>", "}", text_data)

                    part_data = re.finditer(
                        r"\{(.*?)\}", text_data, re.MULTILINE | re.DOTALL
                    )  # take the internal of {...}

                    for lines in part_data:
                        lines = lines.group(1).strip()
                        lines = re.sub(" +", " ", lines)

                        for line in lines.split("\n"):
                            sentence = []
                            for raw_word in line.split(" "):
                                word = preprocess_word(raw_word)
                                if len(word) > 0:
                                    sentence.append(word)
                            if len(sentence) > 0:
                                f_lm.write(" ".join(sentence) + "\n")

    print("Done!", flush=True)
