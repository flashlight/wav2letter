from __future__ import print_function

import os
import sys
from multiprocessing.pool import ThreadPool


CONTRACTIONS = "contractions.txt"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run_for_id(file_name):
    print("Starting thread")
    contractions = []
    with open(CONTRACTIONS, "r") as c:
        for line in c:
            contractions.append(line.strip())

    print("Parsing input file")

    lines = []
    with open(file_name, "r") as f:
        for line in f:
            lines.append(line)

    print("Done parsing input file")

    # with open(file_name + ".filtered", "w") as f:
    filtered_lines = []
    counter = 0
    for line in lines:
        counter += 1
        if counter % 10000 == 0:
            print("Counter at " + str(counter))
        filtered_words = []
        for word in line.strip().split(" "):
            word = word.strip()
            # Take care of cases like "'you'd" or "can't'"
            if word[1:] in contractions:
                filtered_words.append(word[1:])
            elif word[:-1] in contractions:
                filtered_words.append(word[:-1])
            elif word in contractions or "'s" in word:
                filtered_words.append(word)
            else:
                # Check if between two letters
                idx = word.find("'")
                if idx != -1:
                    # Check if apostrophe occurs between two letters (consider valid if so)
                    if idx + 1 < len(word) and idx != 0:
                        filtered_words.append(word)
                    else:
                        filtered_words.append(word.strip().replace("'", ""))
                else:
                    filtered_words.append(word)
        filtered_lines.append(" ".join(filtered_words))

    print("Writing output file")
    with open(file_name + ".filtered", "w") as f:
        for line in filtered_lines:
            f.write(line)
            f.write("\n")


def run():
    # Can be parallelized with a thread pool
    # pool = ThreadPool()
    # pool.map(run_for_id, [os.path.join(sys.argv[1], f) for f in os.listdir(sys.argv[1])])

    run_for_id(os.path.join(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    run()
