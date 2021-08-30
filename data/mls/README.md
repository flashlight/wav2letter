# A Recipe for the MLS corpus.

Multilingual LibriSpeech (MLS) dataset is a large multilingual corpus suitable for speech research. The dataset is derived from read audiobooks from LibriVox and consists of 8 languages - English, German, Dutch, Spanish, French, Italian, Portuguese, Polish. It is available at [OpenSLR](http://openslr.org/94).

## Steps to download and prepare the audio and text data

First download the dataset and untar the file. Replace `[lang]` with appropriate language.

```
wget https://dl.fbaipublicfiles.com/mls/mls_[lang].tar.gz
tar -I pigz -xf mls_[lang].tar.gz
```

Prepare train, dev and test sets as list files to be used for training with wav2letter. Replace `[...]` with appropriate paths

```
python prepare.py -indir [...]/mls_[lang] -outdir [...]
```

The following structure will be generated
```
> tree
.
└── lists
    ├── dev.lst
    ├── test.lst
    └── train.lst
```
