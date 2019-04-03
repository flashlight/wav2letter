This directory contains data processing scripts and training/decoding configs for
performing speech recognition using wav2letter++ on popular datasets.

## Preparing data

Requirements to run scripts:
- python 3
- [`sox`](https://pypi.org/project/sox/) and [`tqdm`](https://pypi.org/project/tqdm/) libraries, available through `pip`.

In general, each dataset contains `prepare_data.py` which prepares the Dataset and Tokens file and `prepare_lm.py` which prepares Lexicon and Language Model data. See `README.md` under each dataset directory for extra dataset-specific setups. Each file in the directory has instruction on how to run the python script.

> [...]/prepare_data.py [OPTIONS ...]

> [...]/prepare_lm.py [OPTIONS ...]

## Training/Decoding

The configs for training and decoding can be found under `configs` folder. Make sure to replace `[...]` with appropriate paths.

To run training
> [...]/wav2letter/build/Train train --flagsfile train.cfg

To run decoding
> [...]/wav2letter/build/Decode --flagsfile decode.cfg


*Replace [...] with appropriate paths*
