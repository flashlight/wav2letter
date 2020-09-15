# [Iterative Pseudo-Labeling for Speech Recognition](https://arxiv.org/abs/2005.09267)

## Abstract
Pseudo-labeling has recently shown promise in end-to-end automatic speech recognition (ASR). We study Iterative Pseudo-Labeling (IPL), a semi-supervised algorithm which efficiently performs multiple iterations of pseudo-labeling on unlabeled data as the acoustic model evolves. In particular, IPL fine-tunes an existing model at each iteration using both labeled data and a subset of unlabeled data. We study the main components of IPL: decoding with a language model and data augmentation. We then demonstrate the effectiveness of IPL by achieving state-of-the-art word-error rate on the Librispeech test sets in both standard and low-resource setting. We also study the effect of language models trained on different corpora to show IPL can effectively utilize additional text. Finally, we release a new large in-domain text corpus which does not overlap with the Librispeech training transcriptions to foster research in low-resource, semi-supervised ASR.

## Gutenberg Language Model
We release a new LM training corpus including abandunt books from [Gutenberg Project](https://www.gutenberg.org/). The corpus is designed for low-resource ASR study with [LibriSpeech](http://www.openslr.org/12) (LS) and [LibriLight](https://ai.facebook.com/tools/libri-light) (LV) datasets by carefully filtering out the potential transcriptions belonging to the training/dev/test data of LibriSpeech and LibriLight.

| LM | Description | Corpus | Vocabulary | Model
|:-:|:-:|:-:|:-:|:-:|
| LS \ LV | Librispeech LM corpus without LV transcriptions | [corpus](https://github.com/facebookresearch/wav2letter/tree/master/recipes/sota/2019#non-overlap-lm-corpus-librispeech-official-lm-corpus-excluded-the-data-from-librivox) | 200K vocab | [lm](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_nooverlap_librispeech_unique_4gram_200kvocab.bin) |
| GB \ LS \ LV | Gutenberg books without LS transcriptions, LV transcriptions | [raw](https://dl.fbaipublicfiles.com/wav2letter/ipl/gutenberg_corpus/gutenberg_intersect_librivox.ids_and_manual.and.librispeech_train-dev-test.ids_and_manual.gutenberg.corpus), [normalized](https://dl.fbaipublicfiles.com/wav2letter/ipl/gutenberg_corpus/gutenberg_intersect_librivox.ids_and_manual.and.librispeech_train-dev-test.ids_and_manual.gutenberg.corpus.normalized) | [200K vocab](https://dl.fbaipublicfiles.com/wav2letter/ipl/gutenberg_lm/gutenberg_intersect_librivox.ids_and_manual.and.librispeech_train-dev-test.ids_and_manual.gutenberg.corpus.normalized.freq.kenlm.200kvocab) | [lm](https://dl.fbaipublicfiles.com/wav2letter/ipl/gutenberg_lm/gutenberg_minus_librivox_minus_librispeech_train-dev-test_5gram_200kvocab.bin) |
| GB \ LV | Gutenberg books without LV transcriptions | [raw](https://dl.fbaipublicfiles.com/wav2letter/ipl/gutenberg_corpus/gutenberg_minus_librivox.ids_and_manual.and.librispeech_dev-test.ids_and_manual.gutenberg.corpus), [normalized](https://dl.fbaipublicfiles.com/wav2letter/ipl/gutenberg_corpus/gutenberg_minus_librivox.ids_and_manual.and.librispeech_dev-test.ids_and_manual.gutenberg.corpus.normalized) | [200K vocab](https://dl.fbaipublicfiles.com/wav2letter/ipl/gutenberg_lm/gutenberg_minus_librivox.ids_and_manual.and.librispeech_dev-test.ids_and_manual.gutenberg.corpus.normalized.freq.kenlm.200kvocab) | [lm](https://dl.fbaipublicfiles.com/wav2letter/ipl/gutenberg_lm/gutenberg_minus_librivox_minus_librispeech_dev-test_5gram_200kvocab.bin) |



## Acoustic Models
We release our pretrained models from the paper. The results in the paper can be reproduced from the models with the following project commits:
- [flashlight](https://github.com/facebookresearch/flashlight) - commit [`e62eb7ea4c9381411508c08226598ba11cbf9511`](https://github.com/facebookresearch/flashlight/commit/e62eb7ea4c9381411508c08226598ba11cbf9511)
- [wav2letter](https://github.com/facebookresearch/wav2letter/) - commit [`d02f08749ce3cf0eeefa4406f61ad9dddb4a19b2`](https://github.com/facebookresearch/wav2letter/commit/d02f08749ce3cf0eeefa4406f61ad9dddb4a19b2)

The architecture of the models can be found in [here](https://github.com/facebookresearch/wav2letter/blob/master/recipes/sota/2019/am_arch/am_transformer_ctc_librivox.arch), which is the best transformer CTC architecture we developed in [End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures](https://arxiv.org/abs/1911.08460).

### Tokens and Lexicons
| Labeled Set | Lexicon | Tokens |
|:-:|:-:|:-:|
| LibriLight-train-10h | [lexicon](https://dl.fbaipublicfiles.com/wav2letter/self_training/librispeech-train-clean-100%2Bdev-unigram-5000-nbest10.dict) | [tokens](https://dl.fbaipublicfiles.com/wav2letter/self_training/librispeech-train-clean-100-unigram-5000.vocab-filtered) |
| LibriSpeech-train-clean-100 | [lexicon](https://dl.fbaipublicfiles.com/wav2letter/self_training/librispeech-train-clean-460%2Bdev-unigram-5000-nbest10.dict) | [tokens](https://dl.fbaipublicfiles.com/wav2letter/self_training/librispeech-train-clean-100-unigram-5000.vocab-filtered) |
| LibriSpeech-train-960h | [lexicon](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/librispeech-train%2Bdev-unigram-10000-nbest10.lexicon) | [tokens](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/librispeech/models/am/librispeech-train-all-unigram-10000.tokens) |

### Pre-trained Models
| Labeled Data | Unlabeled Data | AM: dev-clean | AM: dev-other | LM
|:-:|:-:|:-:|:-:|:-:|
| LL-10 | LS-960 | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/10h_960h_standardlm_rescore_dev_clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/10h_960h_standardlm_rescore_dev_other.bin) | LS \ LV
| LL-10 | LS-960 | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/10h_960h_gblm_rescore_dev_clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/10h_960h_gblm_rescore_dev_other.bin) | GB \ LS \ LV
| LL-10 | LS-960 + LV | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/10h_lv_standardlm_dev_clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/10h_lv_standardlm_dev_other.bin) | LS \ LV
| LL-10 | LS-960 + LV | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/10h_lv_gblm_dev_clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/10h_lv_gblm_dev_other.bin) | GB \ LS \ LV
| Ls-100 | LS-860 | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/100h_960h_standardlm_rescore_dev_clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/100h_960h_standardlm_rescore_dev_other.bin) | LS \ LV
| Ls-100 | LS-860 | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/100h_960h_gblm_rescore_dev_clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/100h_960h_gblm_rescore_dev_other.bin) | GB \ LS \ LV
| Ls-100 | LS-860 + LV | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/100h_lv_standardlm_dev_clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/100h_lv_standardlm_dev_other.bin) | LS \ LV
| Ls-100 | LS-860 + LV | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/100h_lv_gblm_dev_clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/100h_lv_gblm_dev_other.bin) | GB \ LV \ LS
| Ls-960 | LV | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/960h_lv_standardlm_dev_clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/960h_lv_standardlm_dev_other.bin) | LS \ LV
| Ls-960 | LV | [dev-clean](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/960h_lv_gblm_dev_clean.bin) | [dev-other](https://dl.fbaipublicfiles.com/wav2letter/ipl/acoustic_model/960h_lv_gblm_dev_other.bin) | GB \ LV

The LM mentioned in the above table is the one used in IPL training.





## Citation
```
@article{xu2020iterative,
  title={Iterative Pseudo-Labeling for Speech Recognition},
  author={Xu, Qiantong and Likhomanenko, Tatiana and Kahn, Jacob and Hannun, Awni and Synnaeve, Gabriel and Collobert, Ronan},
  journal={arXiv preprint arXiv:2005.09267},
  year={2020}
}
```
