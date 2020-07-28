# Models

This directory contains recipes to reproduce results of training and decoding for approaches published in the papers:
- [NEW] `local_prior_match` [Semi-Supervised Speech Recognition via Local Prior Matching](https://arxiv.org/abs/2002.10336)
- [NEW] `streaming_convnets` [Scaling Up Online Speech Recognition Using ConvNets](https://research.fb.com/publications/scaling-up-online-speech-recognition-using-convnets/)
- [NEW] `sota/2019` [End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures](https://arxiv.org/abs/1911.08460)
- `self_training` [Self-Training for End-to-End Speech Recognition](https://arxiv.org/abs/1909.09116)
- `learnable_frontend` [Learning Filterbanks from Raw Speech for Phone Recognition)](https://arxiv.org/pdf/1711.01161.pdf)
- `seq2seq_tds` [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://arxiv.org/abs/1904.02619)
- `lexicon_free` [Who Needs Words? Lexicon-Free Speech Recognition](https://arxiv.org/abs/1904.04479)
- `conv_glu` [Wav2Letter: an End-to-End ConvNet-based Speech Recognition System](https://arxiv.org/pdf/1609.03193.pdf)

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
