# LM Corpus Reproduction

This document describes normalization of "no overlap" language model corpus, which contains the LibriSpeech LM corpus excluding any books that are contained in LibriVox audio used in the paper, details on corpus creation see in [`../raw_lm_corpus`](../raw_lm_corpus/README.md).

## Data Normalization (Librispeech LM data without Librivox)
```
git clone --recursive https://github.com/moses-smt/mosesdecoder.git
pip install num2words roman
source normalize.sh ../raw_lm_corpus/librispeech_lm_corpus.minus_librivox.metadata_and_manual_and_missing.corpus.txt
python3 generate_uniq.py librispeech_lm_corpus_raw_without_librivox.txt.norm
python3 generate_frequencies.py librispeech_lm_corpus_raw_without_librivox.txt.norm.unique
python3 generate_kenlm_vocab.py librispeech_lm_corpus_raw_without_librivox.txt.norm.unique.freq 200000 200k
```

## Train 4gram LM
```
KENLM=[...]/kenlm/build/bin
devclean=[DATA_PATH]/text/dev-clean.txt
devother=[DATA_PATH]/text/dev-other.txt

"$KENLM/lmplz" -T /tmp -S 100G --discount_fallback -o 4 \
    --limit_vocab_file librispeech_lm_corpus_raw_without_librivox.txt.norm.unique.freq.kenlm.200kvocab trie < librispeech_lm_corpus_raw_without_librivox.txt.norm.unique > nooverlap_librispeech_kenlm_4g_200kvocab.arpa
"$KENLM/build_binary" trie nooverlap_librispeech_kenlm_4g_200kvocab.arpa nooverlap_librispeech_kenlm_4g_200kvocab.bin
"$KENLM/query" nooverlap_librispeech_kenlm_4g_200kvocab.bin < "$devclean" > nooverlap_librispeech_kenlm_4g_200kvocab.bin.dev_clean
"$KENLM/query" nooverlap_librispeech_kenlm_4g_200kvocab.bin < "$devother" > nooverlap_librispeech_kenlm_4g_200kvocab.bin.dev_other
```

## Decoding with 4gram
Running random search with 128 tries over `lmweigth` and `wordscore` parameters with the same config as `decode_pl.cfg` (this config contains best params we found and this config is used to generate pseudo labels). Here we use our best transformer CTC model trained on Librispeech only.

Config `decode_pl_overlap.cfg` is used for ablation study (parameters are from the best config of `../librispeech/decode_transformer_ctc_dev_other.cfg`) - here we change only the language model. Decoding with train only lexicon is very similar to the decoding with 200k lexicon as in the `../librispeech/decode_transformer_ctc_dev_other.cfg`.

To run pseudo labels generation:
```
[...]/wav2letter/build/Decoder --flagsfile decode_pl.cfg --minloglevel=0 --logtostderr=1
```
for ablation study:
```
[...]/wav2letter/build/Decoder --flagsfile decode_pl_overlap.cfg --minloglevel=0 --logtostderr=1
```

We then prepare lists for training and place them to [DATA_DST_librilight]/lists/librivox.lst and [DATA_DST_librilight]/lists/librivox_overlap.lst respectively.

For ablation we also prepared randomly sampled lists of 1000 hour, 3000 hours, and 10000 hours. Each above subset has each smaller one as a subset; i.e. the 1000h set is a subset of the 3000h set, which is a subset of the 10000h set.
