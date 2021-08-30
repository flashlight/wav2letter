import nltk
import tqdm


def load():
    with open("lmtext_no_am.txt", "r") as fid:
        lines = [l.strip() for l in fid]
    return lines


if __name__ == "__main__":
    lines = load()
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    step = 10000
    with open("lmtext_sentences_no_am.txt", "w") as fid:
        for i in tqdm.tqdm(range(0, len(lines), step)):
            sentences = tokenizer.tokenize(" ".join(lines[i : i + step]))
            for l in sentences:
                fid.write(l)
                fid.write("\n")
