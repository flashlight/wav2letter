import argparse
import os
import sys


def run(basefile, tablefile, separator):
    # Read IDs
    with open(basefile, "r") as f:
        titles = [line.strip() for line in f]

    # Make table
    with open(tablefile, "r") as f:
        table = {}
        for line in f:
            book_id, book_title = line.strip().split(separator)
            table[book_title] = book_id

    # Lookup
    for key in titles:
        if key in table:
            sys.stdout.write("{key}|{val}\n".format(key=key, val=table[key]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Join IDs from a table")
    parser.add_argument("--basefile", type=str, required=True)
    parser.add_argument("--tablefile", type=str, required=True)
    parser.add_argument("--separator", type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.basefile):
        raise ValueError("basefile not found")

    if not os.path.exists(args.tablefile):
        raise ValueError("tablefile not found")

    run(args.basefile, args.tablefile, args.separator)
