# Tools

This directory contains tools for audio analysis and processing built on wav2letter.

To build the tools, simply pass `-DW2L_BUILD_TOOLS=ON` as a CMake flag when [building wav2letter](https://github.com/facebookresearch/wav2letter/wiki/General-building-instructions).

## Voice Activity Detection and Audio Analysis

The following scripts perform voice activity detection and basic audio analysis on unlabeled audio. Each script outputs four files named by each input sample ID:
1. A `.vad` file containing chunk-level probabilities of non-speech based on the probability of silence. These are assigned for each chunk of output; for a model trained with a stride of 1, these will be each frame (10 ms), but for a model with a stride of 8, these will be (80 ms) chunks.
2. An `.sts` file containing the perplexity the predicted sequence based on a specified input in addition to the percentage of the audio containing speech based on the passed `--vadthreshold`.
3. A `.tsc` file containing the most likely token-level transcription of given audio based on the acoustic model output only.
4. A `.fwt` file containing frame or chunk-level token emissions based on the most-likely token emitted for each sample.

### Acoustic Models for Audio Analysis

Below are models compatible with the below audio analysis pipelines.

| File | Dataset | Dev Set | Criterion | Architecture | Lexicon | Tokens |
| - | - | - | - | - | - | - |
| [baseline_dev-other](https://dl.fbaipublicfiles.com/wav2letter/audio_analysis/tds_ctc/model.bin) | LibriSpeech | dev-other | CTC | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/audio_analysis/tds_ctc/arch.txt) | [Lexicon](https://dl.fbaipublicfiles.com/wav2letter/audio_analysis/tds_ctc/dict.lst) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/audio_analysis/tds_ctc/tokens.lst) |

## Voice Activity Detection with CTC + an n-gram Language Model
`VoiceActivityDetection-CTC` contains a simple pipeline that supports a CTC-trained acoustic model trained with wav2letter and n-gram language model in an wav2letter binary format (see the [decoder documentation](https://github.com/facebookresearch/wav2letter/wiki/Beam-Search-Decoder) for more).

### Using the Pipeline
Build the tool with `make VoiceActivityDetection-CTC`.

#### Input List File
First, create an input list file containing the audio data. The list file should exactly follow the standard wav2letter [list input format for training](https://github.com/facebookresearch/wav2letter/blob/master/docs/data_prep.md#audio-and-transcriptions-data), but the transcriptions column should be empty. For instance:
```
// Example input file

[~/speech/data] head analyze.lst
train001 /tmp/000000000.flac 100.03
train002 /tmp/000000001.flac 360.57
train003 /tmp/000000002.flac 123.53
train004 /tmp/000000003.flac 999.99
...
...
```

#### Running
Run the binary:
```
[path to binary]/VoiceActivityDetection-CTC \
    -am [path to model] \
    -lm [path to language model] \
    -test [path to list file] \
    --lexicon [path to lexicon file] \
    --maxload -1 \
    --datadir= \
    --tokensdir [path to directory containing tokens file] \
    --tokens [tokens file name] \
    --outpath [output filepath]
```

The output directly specified by outpath will contain.
