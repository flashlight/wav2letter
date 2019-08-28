# Who Needs Words? Lexicon-free Speech Recognition (Likhomanenko et al., 2019)

Below are pre-trained acoustic and language models from [Who Needs Words? Lexicon-free Speech Recognition (Likhomanenko et al., 2019)](https://arxiv.org/abs/1904.04479).

## Acoustic Models
| File | Dataset | Dev Set | Architecture | Lexicon | Tokens |
| - | - | - | - | - | - |
| [baseline_dev-clean+other](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/am/baseline_dev-clean%2Bother.bin) | LibriSpeech | dev-clean+dev-other | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/am.arch) | [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/lexicon.lst) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/tokens.lst) |
| [baseline_nov93dev](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/am/baseline_nov93dev.bin) | WSJ | nov93dev | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/am.arch) | [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/lexicon.lst) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/tokens.lst) |


## Language Models

Convolutional language models (ConvLM) are trained with the [fairseq](https://github.com/pytorch/fairseq) toolkit. n-gram language models are trained with the [KenLM](https://github.com/kpu/kenlm) toolkit. The below language models are converted into a binary format compatible with the wav2letter++ decoder.

| Name |	Dataset | Type | Vocab |
| - | - | - | - |
[lm_librispeech_convlm_char_20B](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_char_20B.bin) | LibriSpeech | ConvLM 20B | [LM Vocab](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_char_20B.vocab)
[lm_librispeech_convlm_word_14B](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_word_14B.bin) | LibriSpeech | ConvLM 14B | [LM Vocab](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_word_14B.vocab)
[lm_librispeech_kenlm_char_15g_pruned](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_kenlm_char_15g_pruned.bin) | LibriSpeech | 15-gram | -
[lm_librispeech_kenlm_char_20g_pruned](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_kenlm_char_20g_pruned.bin) | LibriSpeech | 20-gram | -
[lm_librispeech_kenlm_word_4g_200kvocab](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_kenlm_word_4g_200kvocab.bin) | LibriSpeech | 4-gram | -
[lm_wsj_convlm_char_20B](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_convlm_char_20B.bin) | WSJ | ConvLM 20B | [LM Vocab](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_convlm_char_20B.vocab)
[lm_wsj_convlm_word_14B](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_convlm_word_14B.bin) | WSJ | ConvLM 14B | [LM Vocab](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_convlm_word_14B.vocab)
[lm_wsj_kenlm_char_15g_pruned](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_kenlm_char_15g_pruned.bin) | WSJ | 15-gram | -
[lm_wsj_kenlm_char_20g_pruned](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_kenlm_char_20g_pruned.bin) | WSJ | 20-gram | -
[lm_wsj_kenlm_word_4g](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_kenlm_word_4g.bin) | WSJ | 4-gram | -


## Citation
```
@article{likhomanenko2019needs,
  title={Who needs words? lexicon-free speech recognition},
  author={Likhomanenko, Tatiana and Synnaeve, Gabriel and Collobert, Ronan},
  journal={arXiv preprint arXiv:1904.04479},
  year={2019}
}
```
