# Multilingual LibriSpeech (MLS)

Multilingual LibriSpeech (MLS) dataset is a large multilingual corpus suitable for speech research. The dataset is derived from read audiobooks from LibriVox and consists of 8 languages - English, German, Dutch, Spanish, French, Italian, Portuguese, Polish. It is released on [OpenSLR](http://openslr.org/).

This directory contains pretrained monolingual model releasing and steps for results reproduction.


## Dependencies

- [flashlight](https://github.com/facebookresearch/flashlight)


## Tokens and Lexicons

|  Language  |                                 Token Set                                 |                                    Train Lexicon                                   |                             Joint Lexicon (Train + GB)                             |
|:----------:|:-------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|
|   English  |   [TOKEN](s3://dl.fbaipublicfiles.com/wav2letter/mls/english/tokens.txt)  |   [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/english/train_lexicon.txt)  |   [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/english/joint_lexicon.txt)  |
|   German   |   [TOKEN](s3://dl.fbaipublicfiles.com/wav2letter/mls/german/tokens.txt)   |   [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/german/train_lexicon.txt)   |   [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/german/joint_lexicon.txt)   |
|    Dutch   |    [TOKEN](s3://dl.fbaipublicfiles.com/wav2letter/mls/dutch/tokens.txt)   |    [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/dutch/train_lexicon.txt)   |    [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/dutch/joint_lexicon.txt)   |
|   French   |   [TOKEN](s3://dl.fbaipublicfiles.com/wav2letter/mls/french/tokens.txt)   |   [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/french/train_lexicon.txt)   |   [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/french/joint_lexicon.txt)   |
|   Spanish  |   [TOKEN](s3://dl.fbaipublicfiles.com/wav2letter/mls/spanish/tokens.txt)  |   [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/spanish/train_lexicon.txt)  |   [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/spanish/joint_lexicon.txt)  |
|   Italian  |   [TOKEN](s3://dl.fbaipublicfiles.com/wav2letter/mls/italian/tokens.txt)  |   [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/italian/train_lexicon.txt)  |   [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/italian/joint_lexicon.txt)  |
| Portuguese | [TOKEN](s3://dl.fbaipublicfiles.com/wav2letter/mls/portuguese/tokens.txt) | [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/portuguese/train_lexicon.txt) | [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/portuguese/joint_lexicon.txt) |
|   Polish   |   [TOKEN](s3://dl.fbaipublicfiles.com/wav2letter/mls/polish/tokens.txt)   |   [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/polish/train_lexicon.txt)   |   [Lexicon](s3://dl.fbaipublicfiles.com/wav2letter/mls/polish/joint_lexicon.txt)   |


## Pre-trained acoustic models

|  Language  |                              Architecture                              |                             Acoustic Model                            |
|:----------:|:----------------------------------------------------------------------:|:---------------------------------------------------------------------:|
|   English  |   [Arch](s3://dl.fbaipublicfiles.com/wav2letter/mls/english/arch.txt)  |   [Model](s3://dl.fbaipublicfiles.com/wav2letter/mls/english/am.bin)  |
|   German   |   [Arch](s3://dl.fbaipublicfiles.com/wav2letter/mls/german/arch.txt)   |   [Model](s3://dl.fbaipublicfiles.com/wav2letter/mls/german/am.bin)   |
|    Dutch   |    [Arch](s3://dl.fbaipublicfiles.com/wav2letter/mls/dutch/arch.txt)   |    [Model](s3://dl.fbaipublicfiles.com/wav2letter/mls/dutch/am.bin)   |
|   French   |   [Arch](s3://dl.fbaipublicfiles.com/wav2letter/mls/french/arch.txt)   |   [Model](s3://dl.fbaipublicfiles.com/wav2letter/mls/french/am.bin)   |
|   Spanish  |   [Arch](s3://dl.fbaipublicfiles.com/wav2letter/mls/spanish/arch.txt)  |   [Model](s3://dl.fbaipublicfiles.com/wav2letter/mls/spanish/am.bin)  |
|   Italian  |   [Arch](s3://dl.fbaipublicfiles.com/wav2letter/mls/italian/arch.txt)  |   [Model](s3://dl.fbaipublicfiles.com/wav2letter/mls/italian/am.bin)  |
| Portuguese | [Arch](s3://dl.fbaipublicfiles.com/wav2letter/mls/portuguese/arch.txt) | [Model](s3://dl.fbaipublicfiles.com/wav2letter/mls/portuguese/am.bin) |
|   Polish   |   [Arch](s3://dl.fbaipublicfiles.com/wav2letter/mls/polish/arch.txt)   |   [Model](s3://dl.fbaipublicfiles.com/wav2letter/mls/polish/am.bin)   |


## Pre-trained language models

The `5-gram_lm.arpa` from the tar ball should be used to decode each acoustic model. For faster serialization, people may convert those arpa files into binaries following steps [here](https://kheafield.com/code/kenlm/estimation/).

|  Language  |                            Language Model                            |
|:----------:|:--------------------------------------------------------------------:|
|   English  |   [Model](https://dl.fbaipublicfiles.com/mls/mls_lm_english.tar.gz)  |
|   German   |   [Model](https://dl.fbaipublicfiles.com/mls/mls_lm_german.tar.gz)   |
|    Dutch   |    [Model](https://dl.fbaipublicfiles.com/mls/mls_lm_dutch.tar.gz)   |
|   French   |   [Model](https://dl.fbaipublicfiles.com/mls/mls_lm_french.tar.gz)   |
|   Spanish  |   [Model](https://dl.fbaipublicfiles.com/mls/mls_lm_spanish.tar.gz)  |
|   Italian  |   [Model](https://dl.fbaipublicfiles.com/mls/mls_lm_italian.tar.gz)  |
| Portuguese | [Model](https://dl.fbaipublicfiles.com/mls/mls_lm_portuguese.tar.gz) |
|   Polish   |   [Model](https://dl.fbaipublicfiles.com/mls/mls_lm_polish.tar.gz)   |


## Usage

### Training
```
[...]/flashlight/build/bin/asr/fl_asr_train train --flagsfile=train/<lang>.cfg --minloglevel=0 --logtostderr=1
```

### Decoding
```
[...]/flashlight/build/bin/asr/fl_asr_decode --flagsfile=decode/<lang>.cfg
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
