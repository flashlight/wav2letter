# Steps to reproduce results on Librispeech

## Preparing Text Data and Training the Language Model
In the below section, we show steps to reproduce the creation of the corpus for the language model used at decoding tim, in addition to trainign the ConvLM itself.

### Preparing the Language Model Corpus

The language model corpus used to train uses the standard LibriSpeech LM corpus, but excludes text from all books present in the LibriSpeech audio corpus. The steps that follow exclude these text sources and recreate the corpus used for training.

#### Dependencies

Install dependencies needed for preprocessing steps:

```
pip install nltk
pip install tqdm
```

#### Downloading Assets

Download [LibriSpeech](http://www.openslr.org/12/) metadata (see [`raw-metadata.tar.gz`](http://www.openslr.org/resources/12/raw-metadata.tar.gz)) and the Librispeech LM corpus data to a single directory:
```
wget http://www.openslr.org/resources/12/raw-metadata.tar.gz
```

Grab the [LibriSpeech LM corpus](http://www.openslr.org/11/), in particular, the unnormalized version with the original books (see [`librispeech-lm-corpus.tgz`](http://www.openslr.org/resources/11/librispeech-lm-corpus.tgz)):
```
wget http://www.openslr.org/resources/11/librispeech-lm-corpus.tgz
```

#### Generating, Cleaning, and Preprocessing

Generate Gutenberg book IDs to be removed from the LM corpus. Make sure your current directory has both the metadata and raw LM corpus, i.e:
```
> ls
librispeech-lm-corpus
LibriSpeech
```
Exclude train titles:
```
python3 [path_to_recipe_dir]/lm/generate_lm_raw_text.py
```

Remove numerals and some symbols:
```
sed -i "s/[0-9]*//g" lmtext_sentences_no_am.txt
sed -i "s/\"//g" lmtext_sentences_no_am.txt
```

Remove punctuation and perform further cleaning:
```
python [path_to_recipe_dir]/lm/filter_contractions.py $PATH_TO_CORPUS $CORPUS_FILENAME
```

Further normalization and removal of special characters:
```
python3 clean_lm_text.py
```

Convert to sentences:
```
python3 sentence_ify.py
```

The resulting corpus is the final raw corpus used to train the sentence piece and language model.


## Training a Sentence Piece Model
Run data and auxiliary files (like lexicon, tokens set, etc.) preparation (set necessary paths instead of `[...]`: `dst` path to data list files, `model_dst` path to auxiliary path to store).
```
pip install sentencepiece==0.1.83
python3 prepare_seq2seq_dict.py --src [...] --dst [...]
```
Besides data the auxiliary files for acoustic and language models training/evaluation will be generated:
```
cd $MODEL_DST
tree -L 2
.
└── am
    ├── librispeech-train+dev-unigram-10000.model
    └── librispeech-train-all-unigram-10000.vocab
```

The trained SentencePiece model [is available here](). The vocabulary file [is available here]().

### Generating the Final Corpus

```
# set necessary paths before running
python3 prepare_wp_data.py --data_src [DATA_DST] --model_src [MODEL_DST]
mkdir -p "$MODEL_DST/decoder/fairseq_wp_10k_data/"

python "$FAIRSEQ/preprocess.py" --only-source \
--trainpref "$MODEL_DST/decoder/lm_wp_10k.train" \
--validpref "$MODEL_DST/decoder/lm_wp_10k.dev-clean" \
--testpref "$MODEL_DST/decoder/lm_wp_10k.dev-other" \
--destdir "$MODEL_DST/decoder/fairseq_wp_10k_data/" \
--workers 16
```

Run on the resulting `/checkpoint/jacobkahn/ssl-emergency-lm/lm-data/dict.txt`
