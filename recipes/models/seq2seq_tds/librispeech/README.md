# Steps to reproduce results on Librispeech

## Dependencies
Check out the following commits:
* wav2letter++: https://github.com/facebookresearch/wav2letter/releases/tag/recipes-seq2seq-tds-paper
* flashlight: https://github.com/facebookresearch/flashlight/commit/37266c8a9f270c0fc42546553fd3d150046b2d3b

## Instructions

Run data and auxiliary files (like lexicon, tokens set, etc.) preparation (set necessary paths instead of `[...]`: `data_dst` path to data to store, `model_dst` path to auxiliary path to store).
```
pip install sentencepiece==0.1.82
python3 prepare.py --data_dst [...] --model_dst [...]
```
Besides data the auxiliary files for acoustic and language models training/evaluation will be generated:
```
cd $MODEL_DST
tree -L 2
.
├── am
│   ├── librispeech-train+dev-unigram-10000-nbest10.lexicon
│   └── librispeech-train-all-unigram-10000.tokens
└── decoder
```

To run training/decoding:
- Fix the paths inside `*.cfg`
- Run training with `train.cfg`
The parameters and settings in `train.cfg` are for running experiments on a single node with **8 GPUs** (`--enable_distributed=true`). Distributed jobs can be launched using [Open MPI](https://www.open-mpi.org/).
- Run decoding with `decode*.cfg`

## Pre-trained acoustic and language models

Below there is info about pre-trained acoustic and language models, which one can use, for example, to reproduce a decoding step.

### Acoustic Models
| File | Dataset | Dev Set | Architecture | Lexicon | Tokens |
| - | - | - | - | - | - |
| [baseline_dev-clean](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/models/am/baseline_dev-clean.bin) | LibriSpeech | dev-clean | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/am.arch) | [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/librispeech-train%2Bdev-unigram-10000-nbest10.lexicon) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/librispeech-train-all-unigram-10000.tokens) |
| [baseline_dev-other](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/models/am/baseline_dev-other.bin) | LibriSpeech | dev-other | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/am.arch) | [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/librispeech-train%2Bdev-unigram-10000-nbest10.lexicon) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/librispeech-train-all-unigram-10000.tokens) |

Here architecture files are the same as `network.arch`, tokens and lexicon files generated in the `$MODEL_DST/am/` are the same as in the table.

### Language Models

Convolutional language models (ConvLM) are trained with the [fairseq](https://github.com/pytorch/fairseq) toolkit. n-gram language models are trained with the [KenLM](https://github.com/kpu/kenlm) toolkit. The below language models are converted into a binary format compatible with the wav2letter++ decoder.

| Name |	Dataset | Type | Vocab |
| - | - | - | - |
| [lm_librispeech_convlm_14B](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/models/lm/lm_librispeech_convlm_14B.bin) | LibriSpeech | ConvLM 14B | [LM Vocab](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/models/lm/lm_librispeech_convlm_14B.vocab) |
| [lm_librispeech_kenlm_4g](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/models/lm/lm_librispeech_kenlm_4g.bin) | LibriSpeech | 4-gram | - |

To reproduce decoding step from the paper download these models into `$MODEL_DST/decoder/`.
