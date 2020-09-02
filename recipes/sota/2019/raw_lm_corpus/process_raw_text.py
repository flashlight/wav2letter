import argparse
import os
from multiprocessing.pool import ThreadPool

from gutenberg.cleanup import strip_headers


def strip_header(name):
    print(name)
    with open(name, "r") as intext:
        buf = intext.read().encode("utf-8")
        return strip_headers(buf).strip()


def extract_one_book(book_path):
    content = strip_header(book_path)
    newname = os.path.splitext(book_path)[0] + ".body.txt"
    with open(newname, "w") as outfile:
        outfile.write(content)


def main():
    parser = argparse.ArgumentParser("Processes input Gutenberg text")
    parser.add_argument("--indir", type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.indir):
        raise RuntimeError("indir not found")

    books = [os.path.join(args.indir, f) for f in os.listdir(args.indir)]

    pool = ThreadPool(1)
    pool.map(extract_one_book, books)


if __name__ == "__main__":
    main()
