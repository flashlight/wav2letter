import argparse
import os
import sys
from multiprocessing.pool import ThreadPool

from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_one_book(book_id, outdir):
    eprint("Getting book with id", book_id)
    text = strip_headers(load_etext(book_id)).strip()
    newpath = os.path.join(outdir, str(book_id) + ".body.txt")
    with open(newpath, "w") as outfile:
        outfile.write(text)


def main():
    parser = argparse.ArgumentParser("Grabs Gutenberg books by ID from a file")
    parser.add_argument("--idfile", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.idfile):
        raise RuntimeError("idfile not found")

    with open(args.idfile, "r") as infile:
        ids = [(int(line.strip()), args.outdir) for line in infile]

    pool = ThreadPool(80)
    pool.starmap(get_one_book, ids)


if __name__ == "__main__":
    main()
