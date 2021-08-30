import re
import string
import sys
import unicodedata


LOWER_LETTERS = set(string.ascii_lowercase)
ACCEPTED_LETTERS = set.union(LOWER_LETTERS, set("0123456789"), {"'"})


if __name__ == "__main__":
    for text in sys.stdin:
        # stay only ascii symbols
        nfkd_form = unicodedata.normalize("NFKD", text.strip())
        nfkd_text = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
        # lowercase text
        nfkd_text = nfkd_text.lower()
        # remove hyphen
        nfkd_text = nfkd_text.replace("-", " ")
        # change & -> and
        nfkd_text = nfkd_text.replace("&", " and ")
        nfkd_text = re.sub(" +", " ", nfkd_text).strip()
        # stay words with at least one letter and containing only available tokens
        # otherviwe skip a word
        cleaned_text = []
        for word in nfkd_text.split(" "):
            word = word.lower()
            if len(set(word).intersection(ACCEPTED_LETTERS)) > 0:
                # add word if it contains acceptable tokens
                if len(set(word) - ACCEPTED_LETTERS) == 0:
                    cleaned_text.append(word)
                # add word if last token is . (remove this dot): like words dr., ms., etc
                elif "." in word and len(word) > 1:
                    cleaned_text.append(word)
                else:
                    cleaned_text.append(
                        "".join([letter for letter in word if letter in ACCEPTED_LETTERS])
                    )
            # merge ' for the case ...s'
            elif word == "'":
                if (
                    len(cleaned_text) > 0
                    and len(cleaned_text[-1]) > 1
                    and cleaned_text[-1][-1] == "s"
                    and cleaned_text[-1][-2] != "'"
                ):
                    cleaned_text[-1] += word

        cleaned_text = " ".join(cleaned_text)
        # remove extra whitespaces
        cleaned_text = re.sub(" +", " ", cleaned_text).strip()
        # check if text is empty
        if len(cleaned_text) == 0:
            continue
        # merge '... with its word
        final_text = []
        for word in cleaned_text.split(" "):
            if word[0] != "'":
                final_text.append(word)
            else:
                if len(final_text) > 0:
                    final_text[-1] += word
                else:
                    final_text.append(word)
        print(" ".join(final_text).strip())
