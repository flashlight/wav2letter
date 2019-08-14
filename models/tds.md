# Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions (Hannun et al., 2019)

Below are pre-trained acoustic and language models from [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions (Hannun et al., 2019)](https://arxiv.org/abs/1904.02619).

## Acoustic Models
| File | Dataset | Dev Set | Architecture | Lexicon | Tokens |
| - | - | - | - | - | - |
| [baseline_dev-clean](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/models/am/baseline_dev-clean.bin) | LibriSpeech | dev-clean | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/am.arch) | [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/librispeech-train+dev-unigram-10000-nbest10.lexicon) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/tokens.lst) |
| [baseline_dev-other](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/models/am/baseline_dev-other.bin) | LibriSpeech | dev-other | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/am.arch) | [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/librispeech-train+dev-unigram-10000-nbest10.lexicon) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/librispeech-train-all-unigram-10000.tokens) |


## Language Models

Convolutional language models (ConvLM) are trained with the [fairseq](https://github.com/pytorch/fairseq) toolkit. n-gram language models are trained with the [KenLM](https://github.com/kpu/kenlm) toolkit. The below language models are converted into a binary format compatible with the wav2letter++ decoder.

| Name |	Dataset | Type | Vocab |
| - | - | - | - |
| [lm_librispeech_convlm_14B](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/models/lm/lm_librispeech_convlm_14B.bin) | LibriSpeech | ConvLM 14B | [LM Vocab](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/models/lm/lm_librispeech_convlm_14B.vocab) |
| [lm_librispeech_kenlm_4g](https://dl.fbaipublicfiles.com/wav2letter/tds/librispeech/models/lm/lm_librispeech_kenlm_4g.bin) | LibriSpeech | 4-gram | - |


## Citation
```
@article{DBLP:journals/corr/abs-1904-02619,
  author    = {Awni Hannun and
               Ann Lee and
               Qiantong Xu and
               Ronan Collobert},
  title     = {Sequence-to-Sequence Speech Recognition with Time-Depth Separable
               Convolutions},
  journal   = {CoRR},
  volume    = {abs/1904.02619},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.02619},
  archivePrefix = {arXiv},
  eprint    = {1904.02619},
  timestamp = {Wed, 24 Apr 2019 12:21:25 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1904-02619},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
