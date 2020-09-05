# wav2letter++

[![CircleCI](https://circleci.com/gh/facebookresearch/wav2letter.svg?style=svg)](https://circleci.com/gh/facebookresearch/wav2letter)
[![Join the chat at https://gitter.im/wav2letter/community](https://badges.gitter.im/wav2letter/community.svg)](https://gitter.im/wav2letter/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

wav2letter++ is a [highly efficient](https://arxiv.org/abs/1812.07625) end-to-end automatic speech recognition (ASR) toolkit written entirely in C++, leveraging [ArrayFire](https://github.com/arrayfire/arrayfire) and [flashlight](https://github.com/facebookresearch/flashlight).

The toolkit started from models predicting letters directly from the raw waveform, and now evolved as an all-purpose end-to-end ASR research toolkit, supporting a wide range of models and learning techniques. It also embarks a very efficient modular beam-search decoder, for both structured learning (CTC, ASG) and seq2seq approaches.

**Important disclaimer**: as a number of models from this repository could be used for other modalities, we moved most of the code to flashlight.

This repository includes recipes to reproduce the following research papers as well as **pre-trained** models:
- [NEW] [Pratap et al. (2020): Scaling Online Speech Recognition Using ConvNets](recipes/streaming_convnets/)
- [NEW SOTA] [Synnaeve et al. (2020): End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures](recipes/sota/2019)
- [Kahn et al. (2020): Self-Training for End-to-End Speech Recognition](recipes/self_training)
- [Likhomanenko et al. (2019): Who Needs Words? Lexicon-free Speech Recognition](recipes/lexicon_free/)
- [Hannun et al. (2019): Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](recipes/seq2seq_tds/)

Data preparation for our training and evaluation can be found in [data](data) folder.

The previous iteration of wav2letter can be found in the:
- (before merging codebases for wav2letter and flashlight) [wav2letter-v0.2](https://github.com/facebookresearch/wav2letter/tree/v0.2) branch.
- (written in Lua) [`wav2letter-lua`](https://github.com/facebookresearch/wav2letter/tree/wav2letter-lua) branch.

## Build recipes
First, isntall [flashlight](https://github.com/facebookresearch/flashlight) with all its dependencies. Then
```
mkdir build && cd build && cmake .. && make -j8
```
If flashlight or ArrayFire are installed in nonstandard paths via `CMAKE_INSTALL_PREFIX`, they can be found by passing `-Dflashlight_DIR=[PREFIX]/usr/share/flashlight/cmake/ -DArrayFire_DIR=[PREFIX]/usr/share/ArrayFire/cmake` when running `cmake`.

## Join the wav2letter community
* Facebook page: https://www.facebook.com/groups/717232008481207/
* Google group: https://groups.google.com/forum/#!forum/wav2letter-users
* Contact: vineelkpratap@fb.com, awni@fb.com, qiantong@fb.com, jacobkahn@fb.com, antares@fb.com, avidov@fb.com, gab@fb.com, vitaliy888@fb.com, locronan@fb.com

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
wav2letter++ is BSD-licensed, as found in the [LICENSE](LICENSE) file.
