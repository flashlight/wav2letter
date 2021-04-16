# wav2letter++

[![CircleCI](https://circleci.com/gh/flashlight/wav2letter.svg?style=svg)](https://app.circleci.com/pipelines/github/flashlight/wav2letter)
[![Join the chat at https://gitter.im/wav2letter/community](https://badges.gitter.im/wav2letter/community.svg)](https://gitter.im/wav2letter/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Important Note:
### wav2letter has been moved and consolidated [into Flashlight](https://github.com/flashlight/flashlight) in the [ASR application](https://github.com/flashlight/flashlight/tree/master/flashlight/app/asr).

Future wav2letter development will occur in Flashlight.

*To build the old, pre-consolidation version of wav2letter*, checkout the [wav2letter v0.2](https://github.com/flashlight/wav2letter/releases/tag/v0.2) release, which depends on the old [Flashlight v0.2](https://github.com/flashlight/flashlight/releases/tag/v0.2) release. The [`wav2letter-lua`](https://github.com/flashlight/wav2letter/tree/wav2letter-lua) project can be found on the [`wav2letter-lua` branch](https://github.com/flashlight/wav2letter/tree/wav2letter-lua), accordingly.

For more information on wav2letter++, see or cite [this arXiv paper](https://arxiv.org/abs/1812.07625).

## Recipes
This repository includes recipes to reproduce the following research papers as well as **pre-trained** models:
- [Pratap et al. (2020): Scaling Online Speech Recognition Using ConvNets](recipes/streaming_convnets/)
- [Synnaeve et al. (2020): End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures](recipes/sota/2019)
- [Kahn et al. (2020): Self-Training for End-to-End Speech Recognition](recipes/self_training)
- [Likhomanenko et al. (2019): Who Needs Words? Lexicon-free Speech Recognition](recipes/lexicon_free/)
- [Hannun et al. (2019): Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](recipes/seq2seq_tds/)

Data preparation for training and evaluation can be found in [data](data) directory.

### Building the Recipes

First, install [Flashlight](https://github.com/flashlight/flashlight) with the [ASR application](https://github.com/flashlight/flashlight/tree/master/flashlight/app/asr). Then, after cloning the project source:
```shell
mkdir build && cd build
cmake .. && make -j8
```
If Flashlight or ArrayFire are installed in nonstandard paths via a custom `CMAKE_INSTALL_PREFIX`, they can be found by passing
```shell
-Dflashlight_DIR=[PREFIX]/usr/share/flashlight/cmake/ -DArrayFire_DIR=[PREFIX]/usr/share/ArrayFire/cmake
```
when running `cmake`.

## Join the wav2letter community
* Facebook page: https://www.facebook.com/groups/717232008481207/
* Google group: https://groups.google.com/forum/#!forum/wav2letter-users
* Contact: vineelkpratap@fb.com, awni@fb.com, qiantong@fb.com, jacobkahn@fb.com, antares@fb.com, avidov@fb.com, gab@fb.com, vitaliy888@fb.com, locronan@fb.com

## License
wav2letter++ is BSD-licensed, as found in the [LICENSE](LICENSE) file.
