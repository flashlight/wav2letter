import sys
from collections import defaultdict


def count(MIN_SIL_LENGTH, align_file):
    lines = []
    with open(align_file) as fin:
        lines = fin.readlines()

    res = {}

    res["word_counter"] = [0] * 100  # number of word in each small chunk
    res["chunk_counter"] = [0] * 100  # number of small chunk per audio
    stat = defaultdict(list)
    good_samples = []

    for line in lines:
        sp = line.split("\t")
        # filename = sp[0]

        alignments = sp[1].strip().split("\\n")

        # Parse the alignments
        chunk_starts = [0]
        chunk_ends = []
        words = []

        cur_words = []
        cur_end = 0
        for i, alignment in enumerate(alignments):
            sp = alignment.split()
            begin = float(sp[2])
            length = float(sp[3])
            word = sp[4]

            cur_end = begin + length

            if i == 0:
                continue

            if word == "$":
                if length > MIN_SIL_LENGTH:
                    chunk_ends.append(cur_end)
                    chunk_starts.append(cur_end)
                    words.append(" ".join(cur_words))
                    cur_words = []
                continue

            cur_words.append(word)

        if len(cur_words) > 0:
            chunk_ends.append(cur_end)
            words.append(" ".join(cur_words))
        else:
            chunk_starts.pop()

        # res
        good = True
        n_chunk = len(words)
        # filter if n_segments == 1
        if n_chunk < 2:
            good = False
        res["chunk_counter"][n_chunk] += 1
        for word_chunk in words:
            n_word = len(word_chunk.split())
            res["word_counter"][n_word] += 1
            stat[n_chunk].append(n_word)
            # filter if number of words in a segment > 6
            if n_word > 6:
                good = False
        if good:
            good_samples.append(line)

    print(len(good_samples))
    return res, stat, good_samples


if __name__ == "__main__":
    align_file = sys.argv[1]
    original_file = sys.argv[2]
    res, data, samples = count(0.13, align_file)
    print(res)
    fnames = set([line.strip().split("\t")[0].split("/")[-1] for line in samples])
    # prepare original filtered file
    with open(original_file, "r") as f, open(
        "original.filtered_chunk_g1_ngrams_le6.lst", "w"
    ) as fout:
        for line in f:
            if line.split(" ")[1].split("/")[-1] in fnames:
                fout.write(line)
    with open(align_file + ".filtered_chunk_g1_ngrams_le6", "w") as f:
        for sample in samples:
            f.write(sample)
