from __future__ import absolute_import, division, print_function, unicode_literals

import itertools


class LexiconEntry(object):
    """
    A particular word in the Lexicon and its candidate spellings, sorted by
    """

    def __init__(self, word, sorted_spellings):
        self.word = word
        self.sorted_spellings = sorted_spellings

    def add_spelling(self, spelling):
        self.sorted_spellings.append(spelling)

    def combine_entries(self, other):
        # Zip up sorted spellings
        new_spellings = []
        for this, that in itertools.zip_longest(
            self.sorted_spellings, other.sorted_spellings
        ):
            if this == that:
                new_spellings.append(this)
            else:
                if this:
                    new_spellings.append(this)
                if that:
                    new_spellings.append(that)

        self.sorted_spellings = new_spellings


def write_spellings_to_file(spellings, outfile):
    """
    Writes an array of Spellings to a file in Lexicon format
    """
    sorted_spellings = sorted(spellings, key=lambda spelling: spelling.word)
    with open(outfile, "w") as o:
        for entry in sorted_spellings:
            for spelling in entry.sorted_spellings:
                o.write(entry.word.strip() + " " + " ".join(spelling).strip())
                o.write("\n")


def read_spellings_from_file(infile):
    spellings = {}  # maps string to LexiconEntry
    with open(infile, "r") as infile:
        for line in infile:
            s_idx = line.find(" ")
            word = line[0:s_idx].strip()
            spelling = line[s_idx + 1 :].strip().split(" ")
            if word not in spellings:
                spellings[word] = LexiconEntry(word, [])
            spellings[word].add_spelling(spelling)

    out = []
    for key in sorted(spellings.keys()):
        out.append(spellings[key])
    return out
