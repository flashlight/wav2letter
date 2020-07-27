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
  - supervised LibriSpeech 1k hours + [unsupervised LibriVox 57k hours](https://github.com/facebookresearch/libri-light) (for LibriVox we generate pseudo-labels to use them as a target),
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
- To reproduce acoustic models training on Librispeech (1k hours) and beam-search decoding of these models check the `librispeech` directory.
- Details on pseudolabels preparation is in the directory `lm_corpus_and_PL_generation` (raw LM corpus  which has no intersection with Librovox data is prepared in the `raw_lm_corpus`)
- To reproduce acoustic models training on Librispeech 1k hours + unsupervised LibriVox data (with generated pseudo-labels) and beam-search decoding of these models, check `librivox` directory.
- Details on language models training one can find in the `lm` directory.
- Beam dump for the best models and beam rescoring can be found in the `rescoring` directory.
- Disentangling of acoustic and linguistic representations analyis (TTS and Segmentation experiments) are in `lm_analysis`.

## Tokens and Lexicon sets

| Lexicon | Tokens | Beam-search lexicon | WP tokenizer model |
|:-:|:-:|:-:|:-:|
| [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/librispeech-train%2Bdev-unigram-10000-nbest10.lexicon) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/librispeech-train-all-unigram-10000.tokens) | [Beam-search lexicon](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/decoder-unigram-10000-nbest10.lexicon) | [WP tokenizer model](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/models/am/librispeech-train-all-unigram-10000.model) |

Tokens and lexicon files generated in the `$MODEL_DST/am/` and `$MODEL_DST/decoder/` are the same as in the table.

## Pre-trained acoustic models

Below there is info about pre-trained acoustic models, which one can use, for example, to reproduce a decoding step.

| Dataset | Acoustic model dev-clean | Acoustic model dev-other |
|:-:|:-:|:-:|
| LibriSpeech | [Resnet CTC clean](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_resnet_ctc_librispeech_dev_clean.bin) | [Resnet CTC other](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_resnet_ctc_librispeech_dev_other.bin) |
| LibriSpeech + LibriVox | [Resnet CTC clean](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_resnet_ctc_librivox_dev_clean_icml.bin) | [Resnet CTC other](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_resnet_ctc_librivox_dev_other_icml.bin) |
| LibriSpeech | [TDS CTC clean](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_tds_ctc_librispeech_dev_clean.bin) | [TDS CTC other](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_tds_ctc_librispeech_dev_other.bin) |
| LibriSpeech + LibriVox | [TDS CTC clean](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_tds_ctc_librivox_dev_clean_icml.bin) | [TDS CTC other](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_tds_ctc_librivox_dev_other_icml.bin) |
| LibriSpeech | [Transformer CTC clean](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_transformer_ctc_librispeech_dev_clean.bin) | [Transformer CTC other](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_transformer_ctc_librispeech_dev_other.bin) |
| LibriSpeech + LibriVox | [Transformer CTC clean](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_transformer_ctc_librivox_dev_clean_icml.bin) | [Transformer CTC other](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_transformer_ctc_librivox_dev_other_icml.bin)|
| LibriSpeech | [Resnet S2S clean](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_resnet_s2s_librispeech_dev_clean_icml.bin) | [Resnet S2S other](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_resnet_s2s_librispeech_dev_other_icml.bin) |
| LibriSpeech + LibriVox | [Resnet S2S clean](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_resnet_s2s_librivox_dev_clean_icml.bin) | [Resnet S2S other](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_resnet_s2s_librivox_dev_other_icml.bin) |
| LibriSpeech | [TDS Seq2Seq clean](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_tds_s2s_librispeech_dev_clean.bin) | [TDS Seq2Seq other](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_tds_s2s_librispeech_dev_other.bin)|
| LibriSpeech + LibriVox | [TDS Seq2Seq clean](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_tds_s2s_librivox_dev_clean_icml.bin) | [TDS Seq2Seq other](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_tds_s2s_librivox_dev_other_icml.bin)|
| LibriSpeech | [Transformer Seq2Seq clean](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_transformer_s2s_librispeech_dev_clean.bin) | [Transformer Seq2Seq other](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/am_transformer_s2s_librispeech_dev_other.bin) |
| LibriSpeech + LibriVox | [Transformer Seq2Seq clean](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_transformer_s2s_librivox_dev_clean_icml.bin) | [Transformer Seq2Seq other](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librivox/models/am/am_transformer_s2s_librivox_dev_other_icml.bin) |

## Pre-trained language models

| LM type | Language model | Vocabulary | Architecture | LM Fairseq | Dict fairseq |
|:-:|:-:|:-:|:-:|:-:|:-:|
| ngram | [word 4-gram](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_kenlm_word_4g_200kvocab.bin) | - | - | - | - |
| ngram | [wp 6-gram](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_kenlm_wp_10k_6gram_pruning_000012.bin) | - | - | - | - |
| GCNN | [word GCNN](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_word_14B.bin) | [vocabulary](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_word_14B.vocab) | [Archfile](lm/lm_librispeech_word_gcnn_14B.arch)| [fairseq](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_word_14B.pt) | [fairseq dict](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_word_14B.dict)
| GCNN | [wp GCNN](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_wp_10k_gcnn_14B.bin) | [vocabulary](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_wp_10k_gcnn_14B.vocab) | [Archfile](lm/lm_librispeech_wp_10k_gcnn_14B.arch)| [fairseq](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_wp_10k_gcnn_14B.pt) | [fairseq dict](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_wp_10k_gcnn_14B.dict)
| Transformer | - | - | - | [fairseq](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_word_transformer.pt) | [fairseq dict](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_word_transformer.dict)


**To reproduce decoding step from the paper download these models into `$MODEL_DST/am/` and `$MODEL_DST/decoder/` appropriately.**

## Non-overlap LM corpus (Librispeech official LM corpus excluded the data from Librivox)
One can use prepared corpus to train LM to generate PL on LibriVox data: [raw corpus](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm_corpus/librispeech_lm_corpus.minus_librivox.metadata_and_manual_and_missing.corpus.txt) and [normalized corpus](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm_corpus/librispeech_lm_corpus_raw_without_librivox.txt.norm.unique) and [4gram LM with 200k vocab](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_nooverlap_librispeech_unique_4gram_200kvocab.bin).

## Generated pseudo-labels used in the paper
We open-sourced also the generated pseudo-labels on which we trained our model: [pl](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/pl/librivox.lst) and [pl with overlap](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/pl/librivox-overlap.lst). **Make sure to fix the prefixes to the files names in the lists, right now it is set to be `/root/librivox`)

## Citation
```
@article{synnaeve2019end,
  title={End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures},
  author={Synnaeve, Gabriel and Xu, Qiantong and Kahn, Jacob and Grave, Edouard and Likhomanenko, Tatiana and Pratap, Vineel and Sriram, Anuroop and Liptchinsky, Vitaliy and Collobert, Ronan},
  journal={arXiv preprint arXiv:1911.08460},
  year={2019}
}
```
