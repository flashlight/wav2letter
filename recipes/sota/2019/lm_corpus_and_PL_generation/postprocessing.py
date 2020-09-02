import re
import string
import sys
import unicodedata

import num2words
import roman


LOWER_LETTERS = set(string.ascii_lowercase)
NUMBERS = set("0123456789,")
ROMANS = set("IVXLCDM")
ACCEPTED_LETTERS = set.union(LOWER_LETTERS, {"'"})
PUNCTUATION = set(".,()[]!?")

abbr_mapping = {
    "mr.": "mister",
    "mr": "mister",
    "dr.": "doctor",
    "dr": "doctor",
    "ms.": "miss",
    "ms": "miss",
    "mrs.": "missus",
    "mrs": "missus",
    "vs.": "versus",
    "vs": "versus",
    "st.": "saint",
    "st": "saint",
}


if __name__ == "__main__":
    for text in sys.stdin:
        # stay only ascii symbols
        nfkd_form = unicodedata.normalize("NFKD", text.strip())
        nfkd_text = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
        # remove hyphen
        nfkd_text = nfkd_text.replace("-", " ")
        # change & -> and
        nfkd_text = nfkd_text.replace("&", " and ")
        nfkd_text = re.sub(" +", " ", nfkd_text).strip()
        # stay words with at least one letter and containing only available tokens
        # otherviwe skip a word
        cleaned_text = []
        splitted_text = nfkd_text.split(" ")
        for index, word in enumerate(splitted_text):
            if word == "":
                continue
            # convert roman numbers
            if len(set(word) - ROMANS) == 0 and (
                word != "I"
                or (
                    word == "I"
                    and index > 0
                    and (
                        splitted_text[index - 1] == "Chapter"
                        or splitted_text[index - 1] == "CHAPTER"
                    )
                )
            ):
                try:
                    word = str(roman.fromRoman(word))
                except Exception:
                    pass
            elif (
                len(set(word[:-1]) - ROMANS) == 0
                and len(word) > 1
                and word[-1] in PUNCTUATION
            ):
                try:
                    word = str(roman.fromRoman(word[:-1]))
                except Exception:
                    pass
            # lowercase text
            word = word.lower()
            # process dollars
            if word == "$":
                add_dollar = True
                cleaned_text.append("dollars")
                continue
            if word[0] == "$" and len(word) > 1:
                assert 1 == 0, word
            # preserve numbers
            if len(set(word) - NUMBERS) == 0 and word != ",":
                word = word.replace(",", "")
                if not add_dollar:
                    cleaned_text.append(
                        num2words.num2words(int(word), to="year")
                        .replace(" oh-", " o ")
                        .replace("-", " ")
                        .replace(",", "")
                    )
                if add_dollar:
                    cleaned_text[-1] = (
                        num2words.num2words(int(word), to="cardinal")
                        .replace("-", " ")
                        .replace(",", "")
                    )
                    cleaned_text.append("dollars")
                    add_dollar = False
                continue
            add_dollar = False
            if (
                len(word) > 2
                and len(set(word[:-2]) - NUMBERS) == 0
                and (
                    word[-2:] == "th"
                    or word[-2:] == "st"
                    or word[-2:] == "nd"
                    or word[-2:] == "rd"
                )
            ):
                cleaned_text.append(
                    num2words.num2words(int(word[:-2].replace(",", "")), to="ordinal")
                    .replace("-", " ")
                    .replace(",", "")
                )
                continue
            if word in abbr_mapping:
                cleaned_text.append(abbr_mapping[word])
            elif len(set(word).intersection(LOWER_LETTERS)) > 0:
                # add word if it contains acceptable tokens
                if len(set(word) - ACCEPTED_LETTERS) == 0:
                    cleaned_text.append(word)
                # add word if last token is . (remove this dot): like words dr., ms., etc
                elif len(set(word[:-1]) - ACCEPTED_LETTERS) == 0 and word[-1] == ".":
                    cleaned_text.append(word[:-1])
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
        final_text = " ".join(final_text).strip()
        if final_text == "":
            continue
        print(final_text)
