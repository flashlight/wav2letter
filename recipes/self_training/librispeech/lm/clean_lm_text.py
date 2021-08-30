import re
from multiprocessing import Pool

import nltk
import tqdm


PUNCTUATION = set(list(",'\"?!#&(){}[]*+=;:.-"))
PUNCTUATION.add("")


def clean(line):
    #    try:
    #        new_line = normalise.normalise(line, verbose=False)
    #    except:
    #        print("Could not normalize:", line)
    #    new_line = (t for t in new_line if t not in PUNCTUATION)
    #    new_line = " ".join(new_line).lower()
    new_line = re.sub('[,"?!#&\(\)\{\}\[\]\*+=;:..]', "", line)
    new_line = re.sub("-", " ", new_line)
    return " ".join(new_line.split()).lower()


def write(lines, fid):
    for line in lines:
        if line:
            fid.write(line)
            fid.write("\n")


def load():
    with open("lmtext_sentences_no_am.txt.filtered", "r") as fid:
        lines = [l for l in fid]
    return lines


if __name__ == "__main__":
    lines = load()
    fid = open("lmtext_clean_no_am.txt", "w")
    clean_lines = []
    step = 1000000
    for i in range(0, len(lines), step):
        print("Cleaning lines {} - {}".format(i, i + step))
        pool = Pool()
        clean_lines = pool.map(clean, lines[i : i + step])
        pool.close()
        pool.join()
        write(clean_lines, fid)
    fid.close()
