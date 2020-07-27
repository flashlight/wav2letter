import argparse
import os
import xml.etree.ElementTree as ET
from multiprocessing.pool import ThreadPool

from gutenberg.query import get_metadata


CACHE_PATH = ""


def get_one_title_from_cache(book_id):
    return (book_id, get_metadata("title", int(book_id)))


def get_one_title(book_id):
    try:
        title = (
            ET.parse("{c}/{bid}/pg{bid}.rdf".format(c=CACHE_PATH, bid=str(book_id)))
            .getroot()
            .find(
                "./{http://www.gutenberg.org/2009/pgterms/}ebook/{http://purl.org/dc/terms/}title"
            )
            .text.replace("\n", " ")
            .replace("\r", "")
        )
        return (book_id, title)
    except AttributeError:
        print("Could not get title for book with id", book_id)
        return (book_id, "---DELETE-NO_TITLE")


def main():
    parser = argparse.ArgumentParser(
        "Gets title metadata given a collection of book ids"
    )
    parser.add_argument("--infile", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--cachepath", type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.infile):
        raise ValueError("indir not found")

    if not os.path.exists(args.cachepath):
        raise ValueError("cachepath not found")

    CACHE_PATH = args.cachepath

    book_ids = []
    with open(args.infile, "r") as f:
        book_ids = [line.rstrip() for line in f.readlines()]

    print("Starting thread pool")
    pool = ThreadPool(80)
    id_title_tuples = pool.map(get_one_title, book_ids)

    print("Metadata acquisition complete")

    with open(args.outfile, "w") as o:
        for bid, title in id_title_tuples:
            o.write("{b}|{t}\n".format(b=bid, t=title))


if __name__ == "__main__":
    main()
