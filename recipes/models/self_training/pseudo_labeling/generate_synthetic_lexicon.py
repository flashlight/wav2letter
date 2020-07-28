from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import operator
import os

from synthetic_lexicon_utils import (
    LexiconEntry,
    read_spellings_from_file,
    write_spellings_to_file,
)


def generate_wp_selling(wp_list):
    spellings = []
    this_spelling = []
    for wp in wp_list:
        if not "_" in wp:
            this_spelling.append(wp)
        elif "_" in wp:
            if len(this_spelling) > 0:
                spellings.append(this_spelling)
            this_spelling = [wp]
    if len(this_spelling) > 0:
        spellings.append(this_spelling)
    return spellings


def generate(infile):
    # maps word --> dict mapping wp spellings to the number of
    # times that spelling appears
    lexicon = {}
    with open(infile, "r") as f:
        prediction = None
        wp_spelling_raw = None
        for line in f:
            if "|P|" in line:
                # format is "|P|: _[wp]..."
                prediction = (
                    line[line.find("|P|: ") + len("|P|: ") :]
                    .replace(" ", "")
                    .replace("_", " ")
                )
                continue
            elif "|p|" in line:
                wp_spelling_raw = line[line.find("|p|:") + len("|p|: ") :]
            elif "|T|" in line:
                continue
            elif "|t|" in line:
                continue
            elif "sample" in line:
                continue
            elif "WARNING" in line:
                continue
            elif "CHRONOS" in line:
                continue
            elif "---" in line:
                continue
            else:
                raise Exception("Format invalid; extraneous line: " + line)

            transcription = prediction.strip().split(" ")
            wp_spelling = [e.strip() for e in wp_spelling_raw.strip().split(" ") if e]
            wp_spelling = generate_wp_selling(wp_spelling)

            for transcription_word, wp_spelling_word in zip(transcription, wp_spelling):
                wp_key = " ".join(wp_spelling_word)
                if transcription_word not in lexicon:
                    lexicon[transcription_word] = {}
                if wp_key not in lexicon[transcription_word]:
                    lexicon[transcription_word][wp_key] = 0
                lexicon[transcription_word][wp_key] += 1
    return lexicon


def order_lexicon(lexicon):
    spellings = {}  # maps a transcription word to its spellings, in order
    for transcription_word in lexicon.keys():
        spellings[transcription_word] = []
        for spelling, _freq in sorted(
            lexicon[transcription_word].items(),
            key=operator.itemgetter(1),
            reverse=True,
        ):
            spellings[transcription_word].append(spelling.split(" "))
    return spellings


def create_spellings(spellings):
    entries = {}
    sorted_keys = sorted(spellings.keys())
    for word in sorted_keys:
        for spelling in spellings[word]:
            if word not in entries:
                entries[word] = LexiconEntry(word, [])
            entries[word].add_spelling(spelling)
    return entries


def run():
    parser = argparse.ArgumentParser(
        description="Converts decoder output into train-ready lexicon format"
    )

    parser.add_argument(
        "-i",
        "--inputhyp",
        type=str,
        required=True,
        help="Path to decoder output using --usewordpiece=false file",
    )
    parser.add_argument(
        "-l",
        "--inputlexicon",
        type=str,
        required=True,
        help="Path to the existing lexicon with which to merge a lexicon from the hyp",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to output lexicon file"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.inputhyp):
        raise Exception("'" + args.inputhyp + "' - input file doesn't exist")
    if not os.path.isfile(args.inputlexicon):
        raise Exception("'" + args.inputlexicon + "' - input file doesn't exist")

    lexicon = generate(args.inputhyp)
    sorted_spellings = order_lexicon(lexicon)
    spellings = create_spellings(sorted_spellings)
    new_lexicon = []
    for key in sorted(spellings.keys()):
        new_lexicon.append(spellings[key])

    old_lexicon_spellings = read_spellings_from_file(args.inputlexicon)
    old = {}
    for entry in old_lexicon_spellings:
        old[entry.word] = entry

    count = 0
    for entry in new_lexicon:
        count += 1
        if count % 1000 == 0:
            print("Processed " + str(count) + " entries in new lexicon.")
        if entry.word in old.keys():
            # entry in lexicon, check if spelling exists, else append to end
            for spelling in entry.sorted_spellings:
                if spelling in old[entry.word].sorted_spellings:
                    continue
                else:
                    # only add spelling if we don't already have it
                    if spelling not in old[entry.word].sorted_spellings:
                        old[entry.word].sorted_spellings.append(spelling)
        else:
            # OOV case: create a new lexicon entry with these spellings
            old[entry.word] = entry

    final = []
    # sort the final spellings
    for key in sorted(old.keys()):
        final.append(old[key])

    write_spellings_to_file(final, args.output)


if __name__ == "__main__":
    run()
