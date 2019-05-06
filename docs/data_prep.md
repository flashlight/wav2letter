# Data preparation

For training a Speech Recognition model using wav2letter++, we typically expect the following inputs
- Audio and Transcriptions data
- Token dictionary
- Lexicon
- Language Model

### Audio and Transcriptions data

wav2letter++ expects audio and transcription data to prepared in a specific format so that they can be read from the pipelines.
Each dataset (test/valid/train) needs to be in a separate directory and should contain `.wav` , `.tkn`, `.wrd`, `.id` files
numbered in the following way - 000000000.wav, 000000000.tkn,  000000000.wrd,  000000000.id, 000000001.wav, 000000001.tkn,  000000001.wrd,  000000001.id and so on.
The directories for the dataset is specified using `-datadir`, `-train`, `-valid` and `-test`.

```
[~/speech/data/train] ls | sort | head -n 30
000000000.wav
000000000.id
000000000.tkn
000000000.wrd
000000001.wav
000000001.id
000000001.tkn
000000001.wrd
000000002.wav
000000002.id
000000002.tkn
000000002.wrd
000000003.wav
000000003.id
000000003.tkn
000000003.wrd
000000004.wav
000000004.id
000000004.tkn
000000004.wrd
000000005.wav
000000005.id
000000005.tkn
000000005.wrd
...
...
```

Each sample will have 4 corresponding files
- `.flac/.wav` - audio file. The extension is specified using `-input` flag.
- `.wrd` - words file containing the transcription.
- `.tkn` - tokens file. The extension is specified using `-target` flag.
- `.id `- identifiers for the file. Each line is key-value pair separated by tab

Let's say the transcription for first sample is "hello world" and we are using graphemes as the subword units (tokens).
Here is an example of the contents we would keep in these files

000000000.tkn
```
h e l l o | w o r l d
```

000000000.wrd
```
hello world
```

000000000.id
```
file_id 0
gender  M
speaker_id      3
```

We use [sndfile](https://github.com/erikd/libsndfile/) for loading the audio files.
It supports many different formats which include .wav, .flac etc. and .
For samplerate, 16Khz is the default option but you can specify a different one using `-samplerate` flag.
Note that, we require all the train/valid/test data to have the same format of audio file and the same samplerate for now.
Transcriptions should be specified in `.tkn` and `.wrd` files as mentioned above.

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
