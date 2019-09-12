# Models

This directory contains recipes to reproduce results of training and decoding for approaches published in the papers:
- `conv_glu` [Wav2Letter: an End-to-End ConvNet-based Speech Recognition System](https://arxiv.org/pdf/1609.03193.pdf)
- `seq2seq_tds` [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://arxiv.org/abs/1904.02619)
- `lexicon_free` [Who Needs Words? Lexicon-Free Speech Recognition](https://arxiv.org/abs/1904.04479)

## Steps to reproduce training / decoding

**Note** Make sure to replace `[...], [MODEL_DST], [DATA_DST]` with the appropriate paths in the config files to train and decode.

To run training use
```
[...]/wav2letter/build/Train train --flagsfile train.cfg
```

To run decoding use
```
[...]/wav2letter/build/Decode --flagsfile decode.cfg
```
