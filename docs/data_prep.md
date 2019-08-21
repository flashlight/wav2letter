# Data preparation

For training a Speech Recognition model using wav2letter++, we typically expect the following inputs
- Audio and Transcriptions data
- Token dictionary
- Lexicon
- Language Model

### Audio and Transcriptions data

wav2letter++ expects audio and transcription data to prepared in a specific format
so that they can be read from the pipelines.
Each dataset (test/valid/train) needs to be in a separate file with one sample per line.
A sample is specified using 4 columns separated by space (or tabs).
 - `sample_id` - unique id for the sample
 - `input_handle` - input audio file path.
 - `size` - a real number used for sorting the dataset (typically audio duration).
 - `transcription` - word transcription for this sample

The directories for the dataset is specified using `-datadir` and files are specified
with `-train`, `-valid` and `-test` corresponding to training, validation and test sets.

```
// Example input file format

[~/speech/data] head train.lst
train001 /tmp/000000000.flac 100.03  this is sparta
train002 /tmp/000000001.flac 360.57  coca cola
train003 /tmp/000000002.flac 123.53  hello world
train004 /tmp/000000003.flac 999.99  quick brown fox jumped
...
...
```

We use [sndfile](https://github.com/erikd/libsndfile/) for loading the audio files.
It supports many different formats including .wav, .flac etc.
For samplerate, 16KHz is the default option but you can specify a different one using `-samplerate` flag.
Note that, we require all the train/valid/test data to have the same samplerate for now.

### Token dictionary

A token dictionary file consists of a list of all subword units (graphemes / phonemes /...) used.
If we are using graphemes, a typical token dictionary file would look like this

```
# tokens.txt
|
'
a
b
c
...
... (and so on)
z
```

The symbol "|" is used to denote space.
Note that it is possible to add additional symbols like `N` for noise, `L` for laughter and more depending on the dataset.
If two tokens on the same line in tokens file, they are mapped to the same index for training/decoding.

### Lexicon File

A lexicon file consists of mapping from words to their list of token representation.
Each line will have word followed by its space-split tokens.
Here is an example of grapheme based lexicon.

```
# lexicon.txt
a a |
able  a b l e |
about a b o u t |
above a b o v e |

hello-kitty h e l l o | k i t t y |

...
... (and so on)

```

### Language Model

If a pre-trained LM is not available for the training data,
you can use [KenLM](https://github.com/kpu/kenlm#estimation) for training an N-gram language model, .
It is also recommended to convert arpa files to [binary format](https://github.com/kpu/kenlm#querying) for faster loading.

The wav2letter++ decoder is generic enough to plugin Convolutional LMs, RNN LMs etc. This will be integrated in a later update.
