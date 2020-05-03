# Mixed Precision Training

This directory includes `config` and `architecture` files for training three different models using [Automatic Mixed Precision (AMP)](https://arxiv.org/abs/1710.03740). An analogous approach can be used for training any other model in the mixed precision mode. AMP automatically enables [NVIDIA's Tensor Cores](https://www.nvidia.com/en-us/data-center/tensorcore/) to further accelerate the computations in the compute-intensive parts of a neural network.


## [Conv. GLU Model](https://arxiv.org/pdf/1609.03193.pdf)
Steps to train Conv GLU model on LibriSpeech dataset using AMP:
1. Use [prepare.py](https://github.com/facebookresearch/wav2letter/blob/master/recipes/models/conv_glu/librispeech/prepare.py) and prepare the dataset as instructed [here](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/conv_glu/librispeech).
2. Find `network_amp.arch` and `train_amp.cfg` files in this directory (`mixed_precision/conv_glu/librispeech/`) and use them as explained in what follows to train the network.
2.1. In `train_amp.cfg` set correct path values for `rundir`, `tokensdir`, `archdir`, `train`, `valid`, and `lexicon`.
2.2. If you would like to train on a single GPU, use: 
`./Train train --flagsfile=[PATH]/mixed_precision/conv_glu/librispeech/train_amp.cfg`
Alternatively, you can train a number of GPUs (e.g., 8) using the following command:
`mpirun -n 8 ./Train train -enable_distributed true --flagsfile=[PATH]/mixed_precision/conv_glu/librispeech/train_amp.cfg`


## [Lexicon-free Model](https://arxiv.org/abs/1904.04479)
Steps to train Lexicon-free speech recognition model on LibriSpeech dataset using AMP:
1. Use [prepare.py](https://github.com/facebookresearch/wav2letter/blob/master/recipes/models/lexicon_free/librispeech/prepare.py) and prepare the dataset as instructed [here](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/lexicon_free/librispeech).
2. Find `am_amp.arch` and `train_amp.cfg` files in this directory (`mixed_precision/lexicon_free/librispeech/`) and use them as explained in what follows to train the network.
2.1. In `train_amp.cfg` set correct path values for `rundir`, `tokensdir`, `archdir`, `train`, `valid`, and `lexicon`.
2.2. If you would like to train on a single GPU, use: 
`./Train train --flagsfile=[PATH]/mixed_precision/lexicon_free/librispeech/train_amp.cfg`
Alternatively, you can train a number of GPUs (e.g., 8) using the following command:
`mpirun -n 8 ./Train train -enable_distributed true --flagsfile=[PATH]/mixed_precision/lexicon_free/librispeech/train_amp.cfg`

## [Sequence-to-Sequence TDS Model](https://arxiv.org/abs/1904.02619)
Steps to train Sequence-to-Sequence TDS model on LibriSpeech dataset using AMP:
1. Use [prepare.py](https://github.com/facebookresearch/wav2letter/blob/master/recipes/models/seq2seq_tds/librispeech/prepare.py) and prepare the dataset as instructed [here](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/seq2seq_tds/librispeech).
2. Find `network_amp.arch` and `train_amp.cfg` files in this directory (`mixed_precision/seq2seq_tds/librispeech/`) and use them as explained in what follows to train the network.
2.1. In `train_amp.cfg` set correct path values for `rundir`, `tokensdir`, `archdir`, `train`, `valid`, and `lexicon`.
2.2. If you would like to train on a single GPU, use: 
`./Train train --flagsfile=[PATH]/mixed_precision/seq2seq_tds/librispeech/train_amp.cfg`
Alternatively, you can train a number of GPUs (e.g., 8) using the following command:
`mpirun -n 8 ./Train train -enable_distributed true --flagsfile=[PATH]/mixed_precision/seq2seq_tds/librispeech/train_amp.cfg`