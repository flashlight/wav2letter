import glob
import os


def get_am_bookids():
    ambooks_file = "LibriSpeech/BOOKS.TXT"
    with open(ambooks_file, "r") as fid:
        bookids = [l.split()[0] for l in fid]

    clean_bookids = []
    for bid in bookids:
        try:
            int(bid)
            clean_bookids.append(bid)
        except:
            pass
    return clean_bookids


def load_lm_books():
    lmbook_pattern = "librispeech-lm-corpus/corpus/*/*.txt"
    return glob.glob(lmbook_pattern)


def remove_am_books_from_lm(am_book_ids, lm_books):
    am_book_ids = set(am_book_ids)
    all_lm_books = []
    clean_lm_books = []
    for lmb in lm_books:
        lm_b_id = os.path.splitext(os.path.basename(lmb))[0]
        all_lm_books.append(lm_b_id)
        if lm_b_id not in am_book_ids:
            clean_lm_books.append(lmb)
    all_lm_books = set(all_lm_books)
    for a_id in am_book_ids:
        if a_id not in all_lm_books:
            pass
            # print(a_id)
    return clean_lm_books


def write_lm_books_to_file(lm_books):
    lmfile = "lmtext_no_am.txt"
    with open(lmfile, "w") as fid:
        for lmb in lm_books:
            with open(lmb, "r") as f:
                for line in f:
                    if line.strip() != "":
                        fid.write(line.lower())


if __name__ == "__main__":
    am_book_ids = get_am_bookids()
    lm_books = load_lm_books()
    clean_lm_books = remove_am_books_from_lm(am_book_ids, lm_books)
    print(
        "Removed {} am books from {} lm books. Left with {} lm books".format(
            len(am_book_ids), len(lm_books), len(clean_lm_books)
        )
    )
    # write_lm_books_to_file(lm_books)
