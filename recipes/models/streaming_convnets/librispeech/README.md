# Steps to reproduce results on Librispeech

## Instructions to reproduce

Run data and auxiliary files (like lexicon, tokens set, etc.) preparation (set necessary paths instead of `[...]`: `data_dst` path to data to store, `model_dst` path to auxiliary path to store).
```
pip install sentencepiece==0.1.82
python3 ../../utilities/prepare_librispeech_wp_and_official_lexicon.py --data_dst [...] --model_dst [...] --nbest 10 --wp 10000
```
Besides data the auxiliary files for acoustic and language models training/evaluation will be generated:
```
cd $MODEL_DST
tree -L 2
.
├── am
│   ├── librispeech-train-all-unigram-10000.model
│   ├── librispeech-train-all-unigram-10000.tokens
│   ├── librispeech-train-all-unigram-10000.vocab
│   ├── librispeech-train+dev-unigram-10000-nbest10.lexicon
│   ├── librispeech-train-unigram-10000-nbest10.lexicon
│   └── train.txt
└── decoder
    ├── 4-gram.arpa
    ├── 4-gram.arpa.lower
    └── decoder-unigram-10000-nbest10.lexicon
```

### AM training
- Fix the paths inside `train*.cfg`
- We are running acoustic model training with `train*.cfg` on **32 GPUs** (`--enable_distributed=true` is set in the config).
- Training is done on Librispeech 1k hours and [LibriVox data](https://github.com/facebookresearch/libri-light) (unsupervised for which we used generated pseudo labels from the [End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures](https://arxiv.org/abs/1911.08460)).
```
# Trained during 110 epochs
[...]/wav2letter/build/Train train --flagsfile train_am_500ms_future_context.cfg --minloglevel=0 --logtostderr=1
```

### LM downloading and quantization
```
source prepare_lms.sh [KENLM_PATH]/build/bin [DATA_DST]/decoder
```

### Reproduce beam-search decoding
- Fix the paths inside `decoder*.cfg`
- Run decoding with `decoder*.cfg`
```
[...]/wav2letter/build/Decoder \
  --flagsfile decode_500ms_future_context_ngram_other.cfg \
  --lm=[DATA_DST]/decoder/3-gram.pruned.3e-7.bin.qt \
  --lmweight=0.5515838301157 \
  --wordscore=0.52526055643809

[...]/wav2letter/build/Decoder \
  --flagsfile decode_500ms_future_context_ngram_other.cfg \
  --lm=[DATA_DST]/decoder/3-gram.pruned.3e-7.bin \
  --lmweight=0.51947402167074 \
  --wordscore=0.47301996527186

[...]/wav2letter/build/Decoder \
  --flagsfile decode_500ms_future_context_ngram_other.cfg \
  --lm=[DATA_DST]/decoder/3-gram.pruned.1e-7.bin.qt \
  --lmweight=0.51427799804334 \
  --wordscore=0.17048767049287

[...]/wav2letter/build/Decoder \
  --flagsfile decode_500ms_future_context_ngram_other.cfg \
  --lm=[DATA_DST]/decoder/3-gram.pruned.1e-7.bin \
  --lmweight=0.53898245382313 \
  --wordscore=0.19015993862574

[...]/wav2letter/build/Decoder \
  --flagsfile decode_500ms_future_context_ngram_other.cfg \
  --lm=[DATA_DST]/decoder/3-gram.bin.qt \
  --lmweight=0.67470637680685 \
  --wordscore=0.62867952607587

[...]/wav2letter/build/Decoder \
  --flagsfile decode_500ms_future_context_ngram_other.cfg \
  --lm=[DATA_DST]/decoder/3-gram.bin \
  --lmweight=0.71651725207609 \
  --wordscore=0.83657565205108

[...]/wav2letter/build/Decoder \
  --flagsfile decode_500ms_future_context_ngram_other.cfg \
  --lm=[DATA_DST]/decoder/4-gram.bin.qt \
  --lmweight=0.70340747256672 \
  --wordscore=0.85688768944222

[...]/wav2letter/build/Decoder \
  --flagsfile decode_500ms_future_context_ngram_other.cfg \
  --lm=[DATA_DST]/decoder/4-gram.bin \
  --lmweight=0.71730466678122 \
  --wordscore=0.91529167643869
```

## Pre-trained acoustic models

Below are pre-trained acoustic models which can be used to reproduce results from decoding.

| Dataset | Dev Set | Acoustic model | Architecture | Lexicon | Tokens | Beam-search lexicon | LM model |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| LibriSpeech | dev-other | [500 ms future context](https://dl.fbaipublicfiles.com/wav2letter/streaming_convnets/librispeech/models/am/am_500ms_future_context_dev_other.bin) | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/streaming_convnets/librispeech/am_500ms_future_context.arch) | [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/librispeech-train%2Bdev-unigram-10000-nbest10.lexicon) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/streaming_convnets/librispeech/librispeech-train-all-unigram-10000.tokens) | [Beam-search lexicon](https://dl.fbaipublicfiles.com/wav2letter/streaming_convnets/librispeech/decoder-unigram-10000-nbest10.lexicon) | [LM](https://dl.fbaipublicfiles.com/wav2letter/streaming_convnets/librispeech/models/lm/3-gram.pruned.3e-7.bin.qt)

Here architecture files are the same as `*.arch`, tokens and lexicon files generated in the `$MODEL_DST/am/` and `$MODEL_DST/decoder/` are the same as in the table.

To reproduce decoding step from the paper download these models into `$MODEL_DST/am/` and `$MODEL_DST/decoder/` appropriately.

## Results

| dev-other WER % | test-other WER % | LM model | LM size | Quantized |
|:-:|:-:|:-:|:-:|:-:|
| 7.70 | 8.25 | no LM | - | - |
| 6.75 | 7.48 | 4 gram | 3Gb | N |
| 6.89 | 7.59 | 3 gram | 1.7Gb | N |
| 7.15 | 7.86 | 3 gram prun 1e-7 | 83Mb | N |
| 7.25 | 7.94 | 3 gram prun 3e-7| 36Mb | N |
| 6.75 | 7.50 | 4 gram | 628Mb | Y |
| 6.89 | 7.59 | 3 gram | 349Mb | Y |
| 7.15 | 7.84 | 3 gram prun 1e-7 | 22Mb | Y |
| 7.27 | 7.95 | 3 gram prun 3e-7| 13Mb | Y |
