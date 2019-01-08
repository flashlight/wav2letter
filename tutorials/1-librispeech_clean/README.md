# E2E Speech Recognition on Librispeech-Clean Dataset

This is a step-by-step tutorial on how to build a simple end-to-end speech recognition system using wav2letter++.
We will use "clean" speech subset of [Librispeech](http://www.openslr.org/12) corpus.
The dataset consists of read English speech (sampled at 16KHz) from public domain audio books.

### Step 1: Data preparation

Create an experiment path and download the dataset.

```shell
> W2LDIR=/home/$USER/w2l # or any other path where you want to keep the data
> mkdir -p $W2LDIR
> wget -qO- http://www.openslr.org/resources/12/train-clean-100.tar.gz | tar xvz -C $W2LDIR
> wget -qO- http://www.openslr.org/resources/12/dev-clean.tar.gz | tar xvz -C $W2LDIR
> wget -qO- http://www.openslr.org/resources/12/test-clean.tar.gz | tar xvz -C $W2LDIR
```


After this step, the directory structure of `$W2LDIR` should look like this

```shell
> tree $W2LDIR --filelimit 20
# /home/vineelkpratap/w2l
# └── LibriSpeech
#     ├── BOOKS.TXT
#     ├── CHAPTERS.TXT
#     ├── dev-clean [40 entries exceeds filelimit, not opening dir]
#     ├── LICENSE.TXT
#     ├── README.TXT
#     ├── SPEAKERS.TXT
#     ├── test-clean [40 entries exceeds filelimit, not opening dir]
#     └── train-clean-100 [251 entries exceeds filelimit, not opening dir]
```

`train-clean-100`, `dev-clean`, `test-clean` folders consists of audio files and
their transcripts for train, validation  and test sets respectively.

```shell
> ls $W2LDIR/LibriSpeech/train-clean-100/32/21631/
# 32-21631-0000.flac  32-21631-0003.flac  32-21631-0006.flac  32-21631-0009.flac  32-21631-0012.flac  32-21631-0015.flac  32-21631-0018.flac
# 32-21631-0001.flac  32-21631-0004.flac  32-21631-0007.flac  32-21631-0010.flac  32-21631-0013.flac  32-21631-0016.flac  32-21631-0019.flac
# 32-21631-0002.flac  32-21631-0005.flac  32-21631-0008.flac  32-21631-0011.flac  32-21631-0014.flac  32-21631-0017.flac  32-21631.trans.txt
# `***.trans.txt` in each folder has the transcripts for all the `.flac` audio files present.
```

Now, we will preprocess this dataset into a format which wav2letter++ pipelines can process.

```shell
> wav2letter/tutorials/librispeech_clean/prepare_data.py --src $W2LDIR/LibriSpeech/ --dst $W2LDIR

> wav2letter/tutorials/librispeech_clean/prepare_lm.py --dst $W2LDIR
```

Explore the dataset created

```shell
> tree $W2LDIR --filelimit 20
# ├── data
# │   ├── dev-clean [10812 entries exceeds filelimit, not opening dir]
# │   ├── tokens.txt
# │   ├── test-clean [10480 entries exceeds filelimit, not opening dir]
# │   └── train-clean-100 [114156 entries exceeds filelimit, not opening dir]
# ├── LibriSpeech
# │   ├── BOOKS.TXT
# │   ├── CHAPTERS.TXT
# │   ├── dev-clean [40 entries exceeds filelimit, not opening dir]
# │   ├── LICENSE.TXT
# │   ├── README.TXT
# │   ├── SPEAKERS.TXT
# │   ├── test-clean [40 entries exceeds filelimit, not opening dir]
# │   └── train-clean-100 [251 entries exceeds filelimit, not opening dir]
# └── lm
#     ├── 3-gram.arpa
#     └── lexicon.txt
#
# 9 directories, 8 files
```
You can find more details about dataset preparation [here](../../docs/data_prep.md)

### Step 2: Training the Acoustic Model

During [acoustic model](https://en.wikipedia.org/wiki/Acoustic_model) training, we will train a neural network which learns the relationship between the graphemes and input audio.
We will use Mfsc (a.k.a. logMel) features with 40 filterbanks for this experiment and [Connectionist Temporal Classification](https://distill.pub/2017/ctc/) Loss.
For the neural network, we will use a 8-layer Temporal Convolution blocks with ReLU activations followed by 2 Linear blocks.

To start training

```shell
# Replace [...] with appropriate paths in train.cfg before starting
> [...]/wav2letter/build/Train train --flagsfile [...]/wav2letter/tutorials/librispeech_clean/train.cfg
# ...
# ...
# ...
# ...
# ...
# ...
# I1224 01:19:13.684537 3867118 Train.cpp:443] Epoch 1 started!
# I1224 01:23:47.824731 3867118 Train.cpp:242] epoch:        1 | lr: 0.100000 | lrcriterion: 0.000000 | runtime: 00:04:25 | bch(ms): 37.22 | smp(ms): 0.37 | fwd(ms): 29.52 | crit-fwd(ms): 26.61 | bwd(ms): 5.11 | optim(ms): 1.25 | loss:   27.76703 | train-LER: 64.01 | librispeech/dev-clean-LER: 46.24 | avg-isz: 1267 | avg-tsz: 213 | max-tsz: 400 | hrs:  100.47 | thrpt(sec/sec): 1361.82
# I1224 01:23:48.380589 3867118 Train.cpp:436] Shuffling trainset
# I1224 01:23:48.381211 3867118 Train.cpp:443] Epoch 2 started!
# I1224 01:28:19.144374 3867118 Train.cpp:242] epoch:        2 | lr: 0.100000 | lrcriterion: 0.000000 | runtime: 00:04:22 | bch(ms): 36.73 | smp(ms): 0.35 | fwd(ms): 29.15 | crit-fwd(ms): 26.57 | bwd(ms): 5.02 | optim(ms): 1.16 | loss:   16.05565 | train-LER: 37.99 | librispeech/dev-clean-LER: 37.11 | avg-isz: 1267 | avg-tsz: 213 | max-tsz: 400 | hrs:  100.47 | thrpt(sec/sec): 1380.25
# ...
# ...
# ...
# ...
```

Train the model for 25 epochs and check the run directory (specified by `-rundir`, `-runname` gflags) for logs.
You can also stop training early (lets say after 5 epochs) if you want to proceed quickly to the decoder stage and
not concerned about WER performance.

```shell
> ls [...]/librispeech_clean_trainlogs
# 001_config  001_log  001_model_librispeech#dev-clean.bin  001_model_last.bin  001_perf
```
`001_config` - config used for training

`001_model_librispeech#dev-clean.bin` - saved model for best validation error rate. We'll use this for decoding.

`001_model_last.bin` - model at the end of last epoch

`001_perf` - perf, loss, LER metrics for each epoch

You can find more details about training with wav2letter++ [here](../../docs/train.md) and specifying the architecture files [here](../../docs/arch.md).

### Step 3: Decoding
During decoding, we use lexicon, acoustic model and language model and tune a set of hyperparameters
to get the best word transcription for a given audio file using beam search.

```shell
# Replace [...] with appropriate paths in decode.cfg before starting
> [...]/wav2letter/build/Decode --flagsfile [...]/wav2letter/tutorials/librispeech_clean/decode.cfg
# ...
# ...
# ...
# [Decode data/test-clean (2620 samples) in 199.436s (actual decoding time 0.177s/sample) -- WER: 18.9687, LER: 8.71737]
```

We got a WER of 18.96 on test-clean! You can find more details about decoding with wav2letter++ [here](../../docs/decoder.md)

### Conclusion

In this tutorial, we have trained an end-2-end speech recognition system on "clean" subset of Librispeech dataset.
Here are a few things you can try to explore wav2letter++ further
 - Try 4-gram LM for decoding. Convert the LM from .arpa to [binary format](https://github.com/kpu/kenlm#querying) for faster loading.
 - Tune hyperparams of decoder like beamsize, lmweight etc to get better WER.
 - Use ASG criterion instead of CTC for training the acoustic model.
 - Increase neural network parameters and add dropout to train a better acoustic model.
 - Train the complete end-2-end system on your own dataset.
