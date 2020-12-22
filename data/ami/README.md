# A Recipe for the AMI corpus.

"The AMI Meeting Corpus consists of 100 hours of meeting recordings. The recordings use a range of signals synchronized to a common timeline. These include close-talking and far-field microphones, individual and room-view video cameras, and output from a slide projector and an electronic whiteboard. During the meetings, the participants also have unsynchronized pens available to them that record what is written. The meetings were recorded in English using three different rooms with different acoustic properties, and include mostly non-native speakers." See http://groups.inf.ed.ac.uk/ami/corpus/overview.shtml for more details.

We use the individual headset microphone (IHM) setting for preparing train, dev and test sets. The recipe here is heavily inspired from the preprocessing scripts in Kaldi - https://github.com/kaldi-asr/kaldi/tree/master/egs/ami .

## Steps to download and prepare the audio and text data

Prepare train, dev and test sets as list files to be used for training with wav2letter.  Replace `[...]` with appropriate paths

```
python prepare.py -dst [...]
```

The above scripts download the AMI data, segments them into shorter `.flac` audio files based on word timestamps. Limited supervision training set for 10min, 1hr and 10hr will be generated as well.

The following structure will be generated
```
>tree -L 4
.
├── audio
│   ├── EN2001a
│   │   ├── EN2001a.Headset-0.wav
│   │   ├── ...
│   │   └── EN2001a.Headset-4.wav
│   ├── EN2001b
│   ├── ...
│   ├── ...
│   ├── IS1009d
│   │   ├── ...
│   │   └── IS1009d.Headset-3.wav
│   └── segments
│       ├── ES2005a
│       │   ├── ES2005a_H00_MEE018_0.75_1.61.flac
│       │   ├── ES2005a_H00_MEE018_13.19_16.05.flac
│       │   ├── ...
│       │   └── ...
│       ├── ...
│       └── IS1009d
│            ├── ...
│            └── ...
├── lists
│    ├── dev.lst
│    ├── test.lst
│    ├── train_10min_0.lst
│    ├── train_10min_1.lst
│    ├── train_10min_2.lst
│    ├── train_10min_3.lst
│    ├── train_10min_4.lst
│    ├── train_10min_5.lst
│    ├── train_9hr.lst
│    └── train.lst
│
└── text
    ├── ami_public_manual_1.6.1.zip
    └── annotations
        ├── 00README_MANUAL.txt
        ├── ...
        ├── transcripts0
        ├── transcripts1
        ├── transcripts2
        ├── words
        └── youUsages
```
