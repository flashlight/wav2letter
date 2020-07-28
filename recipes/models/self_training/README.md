# [Self-Training for End-to-End Speech Recognition](https://arxiv.org/abs/1909.09116)

## Abstract
We revisit self-training in the context of end-to-end speech recognition. We demonstrate that training with pseudo-labels can substantially improve the accuracy of a baseline model. Key to our approach are a strong baseline acoustic and language model used to generate the pseudo-labels, filtering mechanisms tailored to common errors from sequence-to-sequence models, and a novel ensemble approach to increase pseudo-label diversity. Experiments on the LibriSpeech corpus show that with an ensemble of four models and label filtering, self-training yields a 33.9% relative improvement in WER compared with a baseline trained on 100 hours of labelled data in the noisy speech setting. In the clean speech setting, self-training recovers 59.3% of the gap between the baseline and an oracle model, which is at least 93.8% relatively higher than what previous approaches can achieve.

## Reproduction
Acoustic model configuration files are provided for each dataset to reproduce results from the paper (training and decoding steps).

Pretrained convolutional language models used in the paper are also included, as well as steps to generate the language model corpus used to train the language models and steps to reproduce acoustic model training.

**Training and decoding broadly follow the existing [TDS seq2seq recipes](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/seq2seq_tds/librispeech).**


### Dependencies

All results from the paper can be reproduced exactly with the following project commits:
- [flashlight](https://github.com/facebookresearch/flashlight) - commit [`77ad2f79249c6833875f57865712de4666617d00`](https://git.io/JvxaN)
- [wav2letter](https://github.com/facebookresearch/wav2letter/) - commit [`57b4904c8c4a808d393f047a9352c2d5be57ae8f`](https://git.io/JvxVa)

Each commit contains versioned documentation for building and installing requisite dependencies.

### Tokens and Lexicon Sets

| Dataset | Unlabeled Set | Lexicon | Tokens |
|:-:|:-:|:-:|:-:|
| LibriSpeech | train-clean-100 Baseline | [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/self_training/librispeech-train-clean-100%2Bdev-unigram-5000-nbest10.dict) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/self_training/librispeech-train-clean-100-unigram-5000.vocab-filtered) |
| LibriSpeech | train-clean-100 + train-clean-360 Oracle | [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/self_training/librispeech-train-clean-460%2Bdev-unigram-5000-nbest10.dict) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/self_training/librispeech-train-clean-100-unigram-5000.vocab-filtered) |
| LibriSpeech | train-clean-100 + train-other-500 Oracle | [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/self_training/librispeech-train-clean-100%2Btrain-other-500%2Bdev-unigram-5000-nbest10.dict) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/self_training/librispeech-train-clean-100-unigram-5000.vocab-filtered) |

Tokens and lexicon files generated in the `$MODEL_DST/am/` and `$MODEL_DST/decoder/` directories following the [LibriSpeech recipe](librispeech/README.md) are the same as in the table.


### Pre-Trained Models
#### Acoustic Models

Components of the baseline model trained only on LibriSpeech training sets are below.

| Dataset | Unlabeled Set | Acoustic Model: dev-clean | Acoustic Model: dev-other |
|:-:|:-:|:-:|:-:|
| LibriSpeech | train-clean-100 Baseline | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/self_training_baseline_tc100_dev-clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/self_training_baseline_tc100_dev-other.bin) |
| LibriSpeech | train-clean-100 + train-clean-360 Oracle | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/self_training_oracle_tc100%2Btc360_dev-clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/self_training_oracle_tc100%2Btc360_dev-other.bin) |
| LibriSpeech | train-clean-100 + train-other-500 Oracle | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/self_training_oracle_tc100%2Bto500_dev-clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/self_training_oracle_tc100%2Bto500_dev-other.bin) |

Below are models trained on pseudo-labels. All train sets include the base train-clean-100 set in addition to generated pseudo-labels. Steps for generating pseudo-labels can be found [here](pseudo_labeling/README.md):

| Dataset | Pseudo-Labeled Set | AM: dev-clean | AM: dev-other | Synthetic Lexicon |
|:-:|:-:|:-:|:-:|:-:|
| LibriSpeech | train-clean-100 + [train-clean-360 PLs](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/pseudo_labels/pseudo-label-decoder-sweep-train-clean-360-new-lm-id10.filters=noeos-ngram4.2-s2sscore.normalized.q9.lst) (single) | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/ssl-tds-s2s-ls-pseudo-label-decoder-sweep-train-clean-360-new-lm-id10.filters=noeos-ngram4.2-s2sscore.normalized.q9-run3_dev-clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/ssl-tds-s2s-ls-pseudo-label-decoder-sweep-train-clean-360-new-lm-id10.filters=noeos-ngram4.2-s2sscore.normalized.q9-run3_dev-other.bin) | [Synthetic Lexicon](https://dl.fbaipublicfiles.com/wav2letter/self_training/synthetic_lexicon/synlex.id5.combined.tc100+dev%2Bdecode_sweep-ls_pseudo-label_train-clean-360-new-lm-id10.lex) |
| LibriSpeech | train-clean-100 + [train-other-500 PLs](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/pseudo_labels/pseudo-label-decoder-sweep-train-other-500-new-lm-id11.filters=noeos-ngram4.2-s2sscore.normalized.q4.lst) (single) | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/ssl-tds-s2s-ls-pseudo-label-decoder-sweep-train-other-500-new-lm-id11.filters=noeos-ngram4.2-s2sscore.normalized.q4-run1_dev-clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/ssl-tds-s2s-ls-pseudo-label-decoder-sweep-train-other-500-new-lm-id11.filters=noeos-ngram4.2-s2sscore.normalized.q4-run1_dev-other.bin) | [Synthetic Lexicon](https://dl.fbaipublicfiles.com/wav2letter/self_training/synthetic_lexicon/synlex.id6.combined.tc100+dev+decode_sweep-ls_pseudo-label_train-other-500-new-lm-id11.lex) |
| LibriSpeech | train-clean-100 + train-clean-360 ([ensemble: 2+3+5+7+8](//dl.fbaipublicfiles.com/wav2letter/self_training/am/pseudo_labels/train-clean-360.ensemble.m2.3.5.7.8.lst)) | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/new_pseudo_train_clean_100_pl360_m23578_r2_dev-clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/self_training/am/new_pseudo_train_clean_100_pl360_m23578_r2_dev-other.bin) |  |

<!-- | LibriSpeech | train-clean-100 + train-other-500 (ensemble) | [dev-clean]() | [dev-other]() | Synthetic Lexicon | -->

#### Language Models

The instructions in [LibriSpeech](librispeech/README.md) contain steps to reproduce the language model training corpus. Below are components of the GCNN language model used for decoding:

| LM type | Language model | Vocabulary | Architecture | LM fairseq | Dict fairseq |
|:-:|:-:|:-:|:-:|:-:|:-:|
| GCNN | [word-piece GCNN](https://dl.fbaipublicfiles.com/wav2letter/self_training/lm/ssl-seq2seq-train_clean_100-wp5k.lm_corpus-minus_train.new.model.bin) | [4k WP](https://dl.fbaipublicfiles.com/wav2letter/self_training/lm/ssl-seq2seq-train_clean_100-wp5k.lm_corpus-minus_train.new.dict.txt) | [Archfile](lm/lm_librispeech_5kwp_gcnn_14B.arch) | [fairseq LM](https://dl.fbaipublicfiles.com/wav2letter/self_training/lm/ssl-seq2seq-train_clean_100-wp5k.lm_corpus-minus_train.new.fairseq_model.pt) | [fairseq Dict](https://dl.fbaipublicfiles.com/wav2letter/self_training/lm/lm_librispeech_word_5kwp_gcnn_14B.dict)

## Citation
```
@article{kahn2019selftraining,
    title={Self-Training for End-to-End Speech Recognition},
    author={Jacob Kahn and Ann Lee and Awni Hannun},
    year={2019},
    eprint={1909.09116},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
