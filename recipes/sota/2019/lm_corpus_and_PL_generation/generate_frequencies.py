import sys
from collections import defaultdict


if __name__ == "__main__":
    words_dict = defaultdict(int)
    path = sys.argv[1]
    out_path = path + ".freq"
    with open(path, "r") as f:
        for line in f:
            for word in line.strip().split():
                words_dict[word] += 1
    with open(out_path, "w") as fout:
        for word, count in sorted(
            words_dict.items(), key=lambda kv: kv[1], reverse=True
        ):
            fout.write("{} {}\n".format(word, count))
