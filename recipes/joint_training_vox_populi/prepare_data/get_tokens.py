# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from common_voice_to_wav2letter import get_base_data_from_csv, PUNCTUATION
from typing import List, Set
import argparse

REPLACE_SET = {"–": "-", "`": "'", "’": "'"}


def get_tokens_from_str(str_in) -> Set[str]:
    for c, val in REPLACE_SET.items():
        str_in = str_in.replace(c, val)
    str_in = str_in.replace(" ", "")
    return {x for x in str_in.lower().replace(" ", "")}


def get_tokens_from_str_list(list_str: List[str]) -> Set[str]:

    out = set()
    for str_in in list_str:
        out = out.union(get_tokens_from_str(str_in))

    return out


def save_tokens(tokens, path_out, eow_token="|") -> None:

    with open(path_out, "w") as f:
        for x in tokens:
            f.write(x + "\n")
        f.write(eow_token)


def main(args):

    data = get_base_data_from_csv(args.input_csv)
    all_tokens = get_tokens_from_str_list([x["text"] for x in data])

    remove_tokens = PUNCTUATION + "…»"
    remove_tokens += "1234567890–"

    all_tokens = all_tokens.difference({x for x in remove_tokens})

    save_tokens(all_tokens, args.output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Token builder")
    parser.add_argument("input_csv")
    parser.add_argument("output")

    main(parser.parse_args())
