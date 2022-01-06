"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import sox


def find_files(src):
    src_dirs = src.split(",")
    required_dirs = [
        "fe_03_p1_sph1",
        "fe_03_p1_sph3",
        "fe_03_p1_sph5",
        "fe_03_p1_sph7",
        "fe_03_p2_sph1",
        "fe_03_p2_sph3",
        "fe_03_p2_sph5",
        "fe_03_p2_sph7",
        "fe_03_p1_sph2",
        "fe_03_p1_sph4",
        "fe_03_p1_sph6",
        "fe_03_p2_sph2",
        "fe_03_p2_sph4",
        "fe_03_p2_sph6",
        "fe_03_p1_tran",
        "fe_03_p2_tran",
    ]
    dir_mapping = {}
    for dir in src_dirs:
        for curdir in os.listdir(dir):
            fulldir = os.path.join(dir, curdir)
            if not os.path.isdir(fulldir):
                continue
            for req_dir in required_dirs:
                new_style_req_dir = req_dir.replace(
                    "fe_03_p1_sph", "fisher_eng_tr_sp_d"
                )
                if curdir == req_dir or curdir == new_style_req_dir:
                    dir_mapping[req_dir] = fulldir
                    continue

    transcript_files = {}
    audio_files = {}
    for dir in required_dirs:
        assert dir in dir_mapping, "could not find the subdirectory {}".format(dir)
        fulldir = dir_mapping[dir]
        if "tran" in fulldir:
            fulldir = os.path.join(fulldir, "data")
        for dirpath, _, filenames in os.walk(fulldir):
            for filename in filenames:
                key = filename.split(".")[0]
                if filename.startswith("fe_") and filename.endswith(".txt"):
                    transcript_files[key] = os.path.join(dirpath, filename)
                elif filename.endswith(".sph"):
                    audio_files[key] = os.path.join(dirpath, filename)

    return [(audio_files[k], transcript_files[k]) for k in audio_files]


def process_fisher_data(sample_data):
    files, _, audio_path, sph2pipe = sample_data
    sphfile, tfile = files
    tmp_files = {}
    for channel in ["A", "B"]:
        tmp_files[channel] = os.path.join(
            audio_path, "{pid}_tmp_{ch}.wav".format(pid=os.getpid(), ch=channel)
        )
        os.system(
            "{sph} -f wav -c {c} {i} {o}".format(
                sph=sph2pipe,
                c=1 if channel == "A" else 2,
                i=sphfile,
                o=tmp_files[channel],
            )
        )
    idx = 0
    lines = []
    with open(tfile, "r") as f:
        first_line = f.readline().strip()
        assert first_line.startswith("#") and first_line.endswith(".sph")
        audiofileid = first_line.replace("#", "").replace(".sph", "").strip()
        cur_audio_path = os.path.join(audio_path, audiofileid)
        os.makedirs(cur_audio_path, exist_ok=True)
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tag, text = line.strip().split(":", 1)
            start, end, channel = tag.split()
            start = float(start)
            end = float(end)
            utt = "{a}-{c}-{s}-{e}".format(
                a=audiofileid,
                c=channel,
                s="{:06d}".format(int(start * 100 + 0.5)),
                e="{:06d}".format(int(end * 100 + 0.5)),
            )

            # ignore uncertain annotations
            if "((" in text:
                continue

            # lower-case
            text = text.lower()

            # remove punctuation
            text = text.replace("?", "")
            text = text.replace(",", "")

            # simplify noise annotations
            text = text.replace("[[skip]]", "")
            text = text.replace("[pause]", "")

            text = text.replace("[laugh]", "[laughter]")

            text = text.replace("[sigh]", "[noise]")
            text = text.replace("[cough]", "[noise]")
            text = text.replace("[mn]", "[noise]")
            text = text.replace("[breath]", "[noise]")
            text = text.replace("[lipsmack]", "[noise]")
            text = text.replace("[sneeze]", "[noise]")

            text = " ".join(text.split())

            out_file = os.path.join(cur_audio_path, "{:09d}.flac".format(idx))
            sox_tfm = sox.Transformer()
            sox_tfm.set_output_format(
                file_type="flac", encoding="signed-integer", bits=16
            )
            sox_tfm.trim(start, end)
            sox_tfm.build(tmp_files[channel], out_file)
            duration = (end - start) * 1000.0
            idx = idx + 1
            lines.append("\t".join([utt, out_file, "{0:.2f}".format(duration), text]))

    # cleanup
    for tmp in tmp_files.values():
        os.remove(tmp)

    return lines
