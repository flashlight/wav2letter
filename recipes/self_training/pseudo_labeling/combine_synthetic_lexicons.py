from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

from synthetic_lexicon_utils import (
    read_spellings_from_file,
    write_spellings_to_file,
)


def combine_lexicons(lexicon1, lexicon2):
    combined = {}
    for lexicon in [lexicon1, lexicon2]:
        for entry in lexicon:
            key = entry.word
            if key in combined:
                combined[key].combine_entries(entry)
            else:
                combined[key] = entry

    combined_list = []
    for key in sorted(combined.keys()):
        combined_list.append(combined[key])
    return combined_list


def run():
    parser = argparse.ArgumentParser(description="Joins two lexicons")

    parser.add_argument(
        "-l1", "--lexicon1", type=str, required=True, help="Path to lexicon 1"
    )
    parser.add_argument(
        "-l2", "--lexicon2", type=str, required=True, help="Path to lexicon 2"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to output lexicon file"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.lexicon1):
        raise Exception("'" + args.lexicon1 + "' - input file doesn't exist")
    if not os.path.isfile(args.lexicon2):
        raise Exception("'" + args.lexicon2 + "' - input file doesn't exist")

    lex1 = read_spellings_from_file(args.lexicon1)
    lex2 = read_spellings_from_file(args.lexicon2)

    combined = combine_lexicons(lex1, lex2)

    write_spellings_to_file(combined, args.output)


if __name__ == "__main__":
    run()
