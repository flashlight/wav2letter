# Multilingual LibriSpeech (MLS)

Multilingual LibriSpeech (MLS) dataset is a large multilingual corpus suitable for speech research. The dataset is derived from read audiobooks from LibriVox and consists of 8 languages - English, German, Dutch, Spanish, French, Italian, Portuguese, Polish. It is available at [OpenSLR](http://openslr.org/94).

This directory contains pretrained monolingual models and also steps to reproduce the results. All the models are trained using 32GB Nvidia V100 GPUs. We have used a total of 64 GPUs for training English, German, Dutch, Spanish, French models and 16 GPUs for training models on Italian, Portuguese and Polish.


## Dependencies

- [flashlight](https://github.com/facebookresearch/flashlight)


## Tokens and Lexicons

|  Language  |                                 Token Set                                 |                                    Train Lexicon                                   |                             Joint Lexicon (Train + GB)                             |
|:----------:|:-------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|
|   English  |   [tokens.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/english/tokens.txt)  |   [train_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/english/train_lexicon.txt)  |   [joint_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/english/joint_lexicon.txt)  |
|   German   |   [tokens.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/german/tokens.txt)   |   [train_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/german/train_lexicon.txt)   |   [joint_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/german/joint_lexicon.txt)   |
|    Dutch   |    [tokens.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/dutch/tokens.txt)   |    [train_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/dutch/train_lexicon.txt)   |    [joint_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/dutch/joint_lexicon.txt)   |
|   French   |   [tokens.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/french/tokens.txt)   |   [train_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/french/train_lexicon.txt)   |   [joint_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/french/joint_lexicon.txt)   |
|   Spanish  |   [tokens.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/spanish/tokens.txt)  |   [train_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/spanish/train_lexicon.txt)  |   [joint_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/spanish/joint_lexicon.txt)  |
|   Italian  |   [tokens.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/italian/tokens.txt)  |   [train_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/italian/train_lexicon.txt)  |   [joint_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/italian/joint_lexicon.txt)  |
| Portuguese | [tokens.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/portuguese/tokens.txt) | [train_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/portuguese/train_lexicon.txt) | [joint_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/portuguese/joint_lexicon.txt) |
|   Polish   |   [tokens.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/polish/tokens.txt)   |   [train_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/polish/train_lexicon.txt)   |   [joint_lexicon.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/polish/joint_lexicon.txt)   |


## Pre-trained acoustic models

|  Language  |                              Architecture                              |                             Acoustic Model                            |
|:----------:|:----------------------------------------------------------------------:|:---------------------------------------------------------------------:|
|   English  |   [arch.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/english/arch.txt)  |   [am.bin](https://dl.fbaipublicfiles.com/wav2letter/mls/english/am.bin)  |
|   German   |   [arch.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/german/arch.txt)   |   [am.bin](https://dl.fbaipublicfiles.com/wav2letter/mls/german/am.bin)   |
|    Dutch   |    [arch.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/dutch/arch.txt)   |    [am.bin](https://dl.fbaipublicfiles.com/wav2letter/mls/dutch/am.bin)   |
|   French   |   [arch.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/french/arch.txt)   |   [am.bin](https://dl.fbaipublicfiles.com/wav2letter/mls/french/am.bin)   |
|   Spanish  |   [arch.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/spanish/arch.txt)  |   [am.bin](https://dl.fbaipublicfiles.com/wav2letter/mls/spanish/am.bin)  |
|   Italian  |   [arch.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/italian/arch.txt)  |   [am.bin](https://dl.fbaipublicfiles.com/wav2letter/mls/italian/am.bin)  |
| Portuguese | [arch.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/portuguese/arch.txt) | [am.bin](https://dl.fbaipublicfiles.com/wav2letter/mls/portuguese/am.bin) |
|   Polish   |   [arch.txt](https://dl.fbaipublicfiles.com/wav2letter/mls/polish/arch.txt)   |   [am.bin](https://dl.fbaipublicfiles.com/wav2letter/mls/polish/am.bin)   |


## Pre-trained language models

The `5-gram_lm.arpa` from the tar ball should be used to decode each acoustic model. For faster loading, people may convert those arpa files into binary format following the steps [here](https://kheafield.com/code/kenlm/estimation/).

|  Language  |                            Language Model                            |
|:----------:|:--------------------------------------------------------------------:|
|   English  |   [mls_lm_english.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_lm_english.tar.gz)  |
|   German   |   [mls_lm_german.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_lm_german.tar.gz)   |
|    Dutch   |    [mls_lm_dutch.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_lm_dutch.tar.gz)   |
|   French   |   [mls_lm_french.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_lm_french.tar.gz)   |
|   Spanish  |   [mls_lm_spanish.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_lm_spanish.tar.gz)  |
|   Italian  |   [mls_lm_italian.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_lm_italian.tar.gz)  |
| Portuguese | [mls_lm_portuguese.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_lm_portuguese.tar.gz) |
|   Polish   |   [mls_lm_polish.tar.gz](https://dl.fbaipublicfiles.com/mls/mls_lm_polish.tar.gz)   |


## Usage

### Preparing the dataset

Follow the steps [here](../../data/mls/) to download and prepare the datset for a given language.

### Training
```
[...]/flashlight/build/bin/asr/fl_asr_train train --flagsfile=train/[lang].cfg --minloglevel=0 --logtostderr=1
```

### Decoding

#### Viterbi
```
[...]/flashlight/build/bin/asr/fl_asr_test --am=[...]/am.bin --lexicon=[...]/train_lexicon.txt --datadir=[...] --test=test.lst --tokens=[...]/tokens.txt --emission_dir='' --nouselexicon --show
```

#### Beam search with language model
```
[...]/flashlight/build/bin/asr/fl_asr_decode --flagsfile=decode/[lang].cfg
```

## Citation

```
@article{Pratap2020MLSAL,
  title={MLS: A Large-Scale Multilingual Dataset for Speech Research},
  author={Vineel Pratap and Qiantong Xu and Anuroop Sriram and Gabriel Synnaeve and Ronan Collobert},
  journal={ArXiv},
  year={2020},
  volume={abs/2012.03411}
}
```

NOTE: We have made few updates to the MLS dataset after our INTERSPEECH paper was submitted to include more number of hours and also to improve the quality of transcripts. To avoid confusion (by having multiple versions), we are making **ONLY** one release with all the improvements included. For accurate dataset statistics and baselines, please refer to the arXiv paper above.
