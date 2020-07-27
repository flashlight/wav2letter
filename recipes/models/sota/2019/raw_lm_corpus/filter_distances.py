import argparse
import os
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run(filename, score, distance_ratio):
    eprint("Starting with filename ", filename)
    with open(filename, "r") as f:
        done = 0
        for line in f:
            done += 1
            str1, str2, scoreRaw = line.split("|")
            distance = float(scoreRaw)
            len1 = len(str1.split())
            len2 = len(str2.split())
            maxlen = max(len1, len2)
            minlen = min(len1, len2)
            if (
                maxlen - minlen
            ) / minlen < distance_ratio and distance <= score * maxlen:
                print("{s1}|{s2}|{d}".format(s1=str1, s2=str2, d=scoreRaw.strip()))
            if done % 1000000 == 0:
                eprint(done)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Filters levenshtein scored title pairs")
    parser.add_argument("--infile", type=str, required=True)
    parser.add_argument("--score", type=float, required=True)
    parser.add_argument("--distance_ratio", type=float, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.infile):
        raise ValueError("infile not found")

    run(args.infile, args.score, args.distance_ratio)
