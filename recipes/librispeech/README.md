A Recipe for the Librispeech corpus.

The Librispeech corpus consists of about 1000 hours of read English speech. The Librispeech corpus can be downloaded for free from [here](http://www.openslr.org/12)

## Preparing data
Extra requirements for running seq2seq experiments:
- [`sentencepiece`](https://github.com/google/sentencepiece) library, available through `pip`.

> [...]/prepare_data.py [OPTIONS ...]

> (optional, for seq2seq experiments only) [...]/prepare_seq2seq_dict.py [OPTIONS ...]

> [...]/prepare_lm.py [OPTIONS ...]

## Training/Decoding

Two configs are available under the `configs` folder. See `README.md` under each folder for more config-specific instructions.
- `conv_glu`: architecture & flags for training models as described in the paper "[Wav2Letter: an End-to-End ConvNet-based Speech Recognition System](https://arxiv.org/pdf/1609.03193.pdf)"

- `seq2seq_tds`: architecture & flags for training models as described in the paper "Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions"
