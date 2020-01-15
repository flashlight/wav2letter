# A Recipe for the Librispeech corpus.

The Librispeech corpus consists of about 1000 hours of read English speech. The Librispeech corpus can be downloaded for free from [here](http://www.openslr.org/12)


## Wav2Letter models
| DEV-CLEAN WER % / LER % | DEV-OTHER WER % / LER % | TEST-CLEAN WER % / LER % | TEST-OTHER WER % / LER % | MODEL                                                                                                      | PAPER                                                                                                     |
|:-------------------------:|:-------------------------:|:--------------------------:|:--------------------------:|:------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|
| - | 3.52 / - | - | 4.11 / - | [SOTA 2019](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/sota/2019)| [End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures](https://arxiv.org/abs/1911.08460)|
| - | 7.27 / - | - | 7.95 / - | [streaming_convnets](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/streaming_convnets)|[Scaling Up Online Speech Recognition Using ConvNets](https://research.fb.com/publications/scaling-up-online-speech-recognition-using-convnets/)|
| 3.01 / - | 8.86 / - | 3.28 / - | 9.84 / - | [seq2seq_tds](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/seq2seq_tds/librispeech)|[Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://arxiv.org/abs/1904.02619)|
| 4.6 / 2.3 | 13.8 / 9.0 | 4.8 / - | 14.5 / - | [conv_glu](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/conv_glu/librispeech) | [Wav2Letter: an End-to-End ConvNet-based Speech Recognition System](https://arxiv.org/pdf/1609.03193.pdf), [Letter-Based Speech Recognition with Gated ConvNets](https://arxiv.org/pdf/1712.09444.pdf) |

## Steps to download and prepare the audio and text data
To download and prepare the audio and text for training/evaluation run (replace [...] with a path where the data should be stored)
```
python3 prepare.py --dst [...]
```

The following structure will be generated
```
tree -L 3
.
├── audio
│   ├── dev-clean.tar
│   ├── dev-other.tar
│   ├── LibriSpeech
│   │   ├── BOOKS.TXT
│   │   ├── CHAPTERS.TXT
│   │   ├── dev-clean
│   │   ├── dev-other
│   │   ├── LICENSE.TXT
│   │   ├── README.TXT
│   │   ├── SPEAKERS.TXT
│   │   ├── test-clean
│   │   ├── test-other
│   │   ├── train-clean-100
│   │   ├── train-clean-360
│   │   └── train-other-500
│   ├── test-clean.tar
│   ├── test-other.tar
│   ├── train-clean-100.tar
│   ├── train-clean-360.tar
│   └── train-other-500.tar
├── lists
│   ├── dev-clean.lst
│   ├── dev-other.lst
│   ├── test-clean.lst
│   ├── test-other.lst
│   ├── train-clean-100.lst
│   ├── train-clean-360.lst
│   └── train-other-500.lst
└── text
    ├── dev-clean.txt
    ├── dev-other.txt
    ├── librispeech-lm-norm.txt
    ├── librispeech-lm-norm.txt.lower.shuffle
    ├── test-clean.txt
    ├── test-other.txt
    ├── train-clean-100.txt
    ├── train-clean-360.txt
    └── train-other-500.txt
```
