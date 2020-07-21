from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audacity Label converter")
    parser.add_argument(
        "-f", "--alignment_file", help="alignment file", default="./alignment.txt"
    )
    parser.add_argument(
        "-o", "--output_dir", help="output_dir", default="."
    )
    args = parser.parse_args()

    with open(args.alignment_file, "r") as f:
        for line in f:
            (sampleId, segments) = line.split('\t')
            segments = segments.split('\\n')
            output_file = os.path.join(args.output_dir, sampleId)
            with open(output_file, "w") as ofile:
                for segment in segments:
                    (_, _, start, duration, word) = segment.split(' ')
                    end = float(start) + float(duration)
                    ofile.write('{}\t{}\t{}\n'.format(start, end, word))
