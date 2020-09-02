"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to prepare unpaired data for training a model with local prior matching

Command : python3 prepare_unpaired.py --data_dst [...] --model_dst [...]

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librispeech Dataset creation.")
    parser.add_argument(
        "--data_dst", help="data destination directory", default="./librispeech"
    )
    parser.add_argument(
        "--model_dst",
        help="model auxilary files destination directory",
        default="./lpm_librispeech",
    )

    args = parser.parse_args()

    subpaths = {
        "unpaired": ["train-clean-360", "train-other-500"],
    }

    lists_path = os.path.join(args.data_dst, "lists")
    am_path = os.path.join(args.model_dst, "am")
    unpaired_lists_path = os.path.join(args.model_dst, "lpm_data")

    reflen_dict = set()
    for name in subpaths["unpaired"]:
        unpaired_data = {}
        with open(os.path.join(lists_path, name + ".lst"), "r") as flist:
            for line in flist:
                file_tag, audio_path, audio_length, _ = line.strip().split(" ", 3)
                unpaired_data[file_tag] = (audio_path, audio_length)

        with open(
            os.path.join(unpaired_lists_path, name + "-viterbi.out"), "r"
        ) as fdata:
            with open(
                os.path.join(unpaired_lists_path, name + "-lpm.lst"), "w"
            ) as fout:
                for line in fdata:
                    file_tag, reflen = line.strip().split(" ", 1)
                    fout.write(
                        "%s %s %s %s\n" % (
                            file_tag,
                            unpaired_data[file_tag][0],
                            unpaired_data[file_tag][1],
                            reflen
                        )
                    )
                    reflen_dict.add(reflen)

    # append reflen* to the new lexicon
    orig_lexicon = "librispeech-paired-train+dev-unigram-5000-nbest10.lexicon"
    lpm_lexicon = \
        "librispeech-paired-train-unpaired-viterbi+dev-unigram-5000-nbest10.lexicon"

    with open(os.path.join(am_path, lpm_lexicon), "w") as fout:
        with open(os.path.join(am_path, orig_lexicon), "r") as fin:
            for line in fin:
                fout.write(line)

        for r in reflen_dict:
            # r's format is "reflen1", "reflen2", ... "reflen100", etc.
            fout.write(r + "\t" + " ".join(["a"] * int(r[6:])) + "\n")

    print("Done!", flush=True)
