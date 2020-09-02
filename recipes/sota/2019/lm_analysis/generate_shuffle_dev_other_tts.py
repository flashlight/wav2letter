import os
import sys

import numpy


numpy.random.seed(42)


with open(os.path.join(sys.argv[1], "dev-other.lst"), "r") as f:
    data = [line.strip() for line in f]

for n, seed_val in enumerate([0, 2, 3, 4, 5]):
    numpy.random.seed(42 + seed_val)
    data = numpy.random.permutation(data)

    with open("tts_shuffled_{}.txt".format(n), "w") as fout:
        for line in data:
            line_new = line.split(" ")
            new_tr = numpy.random.permutation(line_new[3:])
            fout.write(line + "\n")
            fout.write("{}\n".format(" ".join(new_tr)))
