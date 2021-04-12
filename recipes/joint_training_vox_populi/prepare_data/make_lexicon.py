# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Set


log = logging.getLogger(__name__)


def has_valid_tokens(word: str, tokens: Set[str]) -> bool:
    for c in word:
        if c not in tokens:
            return False
    return True


def read_token_file(path_token_file: Path, eow_char: str) -> Set[str]:

    with path_token_file.open("r") as file:
        data = [x.strip() for x in file.readlines()]

    return {x for x in data if x != eow_char}


def save_lexicon(
    lexicon: Set[str], path_out: Path, eow_char: str, tokens: Set[str]
) -> None:

    list_lexicon = list(lexicon)
    list_lexicon.sort()

    with path_out.open("w") as file:
        for word in list_lexicon:
            if has_valid_tokens(word, tokens):
                split_word = " ".join(list(word)) + " " + eow_char + " "
                file.write(f"{word} {split_word}\n")


def load_lexicon(path_lexicon: Path) -> Dict[str, str]:
    with open(path_lexicon, "r") as file:
        data = [x.strip() for x in file.readlines()]

    out = {}
    for line in data:
        word = line[0]
        spelling = " ".join(line[1:])
        out[word] = spelling
    return out


def load_words_from_lst(path_lst: Path, n_best: int, min_occ: int, is_raw_text: bool):
    """
    Load words from an input file, which can be in w2l list format or
    a file with lines of sentences.

    paht_lst: input file
    n_best: top n frequent words to keep
    min_occ: minimum number of occurrences of each word
    is_raw_text: the input file only contains lines of text (True);
                the input file is in w2l list format, including utterance ids and audio path (False)
    """

    with path_lst.open("r") as file_lst:
        data = [x.strip() for x in file_lst.readlines()]

    log.info("Building the lexicon")

    out = {}
    # id_ path duration normalized_text
    for line in data:
        if is_raw_text:
            words = line.split()
        else:
            words = line.split()[3:]
        for word in words:
            if word not in out:
                out[word] = 0

            out[word] += 1

    tmp = list(out.items())
    tmp = [(k, v) for k, v in tmp if v >= min_occ]
    tmp.sort(reverse=True, key=lambda x: x[1])
    return {x for x, v in tmp[:n_best]}


def lexicon_from_lst(
    path_lst: Path,
    path_tokens: Path,
    eow_char: str,
    path_out: Path,
    path_old_lexicon: Optional[Path] = None,
    n_best: int = 5000,
    min_occ: int = 10,
    is_raw_text: bool = False,
) -> None:

    out_lexicon = set()
    tokens = read_token_file(path_tokens, eow_char)
    log.info("Token file loaded")
    out_lexicon = load_words_from_lst(path_lst, n_best, min_occ, is_raw_text)

    if path_old_lexicon is not None:
        old_lexicon = load_lexicon(path_old_lexicon)
        out_lexicon |= old_lexicon.keys()

    log.info(f"Saving the lexicon at {path_out}")
    save_lexicon(out_lexicon, path_out, eow_char, tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build a lexicon from the given .lst file")
    parser.add_argument(
        "-i", "--input_lst", type=str, required=True, help="Path to the input lst file"
    )
    parser.add_argument(
        "--tokens", type=str, required=True, help="Path to the token file"
    )
    parser.add_argument(
        "--max_size_lexicon",
        type=int,
        help="Number of words to retain.",
        default=10000,
    )
    parser.add_argument(
        "--min_occ",
        type=int,
        help="Number of words to retain.",
        default=0,
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to the output file."
    )
    parser.add_argument("--eow_token", type=str, default="|", help="End of word token.")
    parser.add_argument(
        "--old_lexicon",
        type=str,
        default=None,
        help="Add the given lexicon to the output file",
    )
    parser.add_argument(
        "--raw_text", action="store_true", help="input is raw text instead of w2l list"
    )

    args = parser.parse_args()
    path_old_lexicon = Path(args.old_lexicon) if args.old_lexicon is not None else None
    lexicon_from_lst(
        Path(args.input_lst),
        Path(args.tokens),
        args.eow_token,
        Path(args.output),
        path_old_lexicon=path_old_lexicon,
        n_best=args.max_size_lexicon,
        min_occ=args.min_occ,
        is_raw_text=args.raw_text,
    )
