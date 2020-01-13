# [End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures](https://arxiv.org/abs/1911.08460)

In the paper we are considering:
- different architectures for acoustic modeling:
  - ResNet
  - TDS
  - Transformer
- different criterions:
  - Seq2Seq
  - CTC
- different settings:
  - supervised LibriSpeech 1k hours
  - supervised LibriSpeech 1k hours + [unsupervised LibriVox 57k hours](https://github.com/facebookresearch/libri-light) (for LibriVox we generate pseudo labels to use them as a target),
- and different language models:
  - word-piece (ngram, ConvLM)
  - word-based (ngram, ConvLM, transformer)

## Data preparation

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

## Instructions to reproduce training and decoding
- Detailed language models recipes one can find in the `lm` directory.
- To reproduce acoustic models training on Librispeech (1k hours) please go to the `librispeech` directory.
- For models trained on Librispeech 1k hours and unsupervised Librilight data (with generated pseudo labels) we release for now models themselves, arch files and train config (full details are coming soon), check `librivox` directory.
- Rescoring steps are also coming soon (with Transformer language model for rescoring).

### Beam-search decoding
- Fix the paths inside `decode*.cfg`
- Run decoding with `decode*.cfg`
```
[...]/wav2letter/build/Decoder --flagsfile path/to/necessary/decode/config --minloglevel=0 --logtostderr=1
```

## Tokens and Lexicon sets

| Lexicon | Tokens | Beam-search lexicon |
|:-:|:-:|:-:|
| [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/librispeech-train+dev-unigram-10000-nbest10.lexicon) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/librispeech-train-all-unigram-10000.tokens) | [Beam-search lexicon](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/decoder-unigram-10000-nbest10.lexicon) |

Tokens and lexicon files generated in the `$MODEL_DST/am/` and `$MODEL_DST/decoder/` are the same as in the table.

## Pre-trained acoustic models

Below there is info about pre-trained acoustic models, which one can use, for example, to reproduce a decoding step.

| Dataset | Acoustic model dev-clean | Acoustic model dev-other | Architecture |
|:-:|:-:|:-:|:-:|
| LibriSpeech | [Resnet CTC](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_resnet_ctc_librispeech_dev_clean.bin) | [Resnet CTC](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_resnet_ctc_librispeech_dev_other.bin) | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/am/am_resnet_ctc.arch)|
| LibriSpeech + LibriVox | [Resnet CTC](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_resnet_ctc_librivox_dev_clean.bin) | [Resnet CTC](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_resnet_ctc_librivox_dev_other.bin) | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/am/am_resnet_ctc.arch)|
| LibriSpeech | [TDS CTC](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_tds_ctc_librispeech_dev_clean.bin) | [TDS CTC](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_tds_ctc_librispeech_dev_other.bin) | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/am/am_resnet_ctc.arch)|
| LibriSpeech + LibriVox | [TDS CTC](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_tds_ctc_librivox_dev_clean.bin) | [TDS CTC](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_tds_ctc_librivox_dev_other.bin) | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/am/am_tds_ctc_librivox.arch) |
| LibriSpeech + LibriVox | - | [Transformer CTC](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_transformer_ctc_librivox_dev_other.bin) | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/am/am_transformer_ctc.arch)|
| LibriSpeech | [TDS Seq2Seq](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_tds_s2s_librispeech_dev_clean.bin) | [TDS Seq2Seq](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_tds_s2s_librispeech_dev_other.bin) | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/am/am_tds_s2s.arch) |

Here architecture files are the same as `*.arch`,

## Pre-trained language models

| LM type | Language model | Vocabulary | Architecture |
|:-:|:-:|:-:|:-:|
| ngram | [word 4-gram](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_kenlm_word_4g_200kvocab.bin) | - | - |
| ngram | [wp 6-gram](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_kenlm_wp_10k_6gram_pruning_000012.bin) | - | - |
| GCNN | [word GCNN](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_word_14B.bin) | [vocabulary](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_word_14B.vocab) | [Archfile](lm/lm_librispeech_word_gcnn_14B.arch)|
| GCNN | [wp GCNN](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_wp_10k_gcnn_14B.bin) | [vocabulary](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_wp_10k_gcnn_14B.vocab) | [Archfile](lm/lm_librispeech_wp_10k_gcnn_14B.arch)|


**To reproduce decoding step from the paper download these models into `$MODEL_DST/am/` and `$MODEL_DST/decoder/` appropriately.**

## Results

| Data | Model | dev-clean WER % | test-clean WER % | dev-other WER % | test-other WER % | LM |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Librispeech | CTC resnet | 3.93 | 4.15 | 10.13 | 10.10 | - |
| Librispeech | CTC resnet | 3.29 | 3.68 | 8.56 | 8.69 | word 4-gram |
| Librispeech + LibriVox | CTC resnet | 3.08 | 3.42 | 7.80 | 8.23 | - |
| Librispeech + LibriVox | CTC resnet | 2.89 | 3.27 | 6.97 | 7.52 | word 4-gram |
| Librispeech | CTC TDS | 4.22 | 4.69 | 11.16 | 11.21 | - |
| Librispeech | CTC TDS | 3.49 | 3.98 | 9.18 | 9.53 | word 4-gram |
| Librispeech + LibriVox | CTC TDS | 3.08 | 3.42 | 7.92 | 8.25 | - |
| Librispeech + LibriVox | CTC TDS | 2.87 | 3.38 | 7.22 | 7.63 | word 4-gram |
| Librispeech + LibriVox | CTC Transformer | - | - | 6.10 | 6.54 | - |
| Librispeech + LibriVox | CTC Transformer | - | - | 5.69 | 6.18 | word 4-gram |
| Librispeech | Seq2Seq TDS | 3.20 | 3.56 | 8.20 | 8.40 | - |
| Librispeech | Seq2Seq TDS | 2.76 | 3.18 | 7.01 | 7.16 | wp 6-gram |

Other models, decoding with GCNN and rescoring are coming soon.

## Citation
```
@article{synnaeve2019end,
  title={End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures},
  author={Synnaeve, Gabriel and Xu, Qiantong and Kahn, Jacob and Grave, Edouard and Likhomanenko, Tatiana and Pratap, Vineel and Sriram, Anuroop and Liptchinsky, Vitaliy and Collobert, Ronan},
  journal={arXiv preprint arXiv:1911.08460},
  year={2019}
}
```
