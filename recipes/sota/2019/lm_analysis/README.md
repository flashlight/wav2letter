# Disentangling acoustic and linguistic representaitons

## TTS experiments

### Set environment
Follow instructions here https://github.com/mozilla/TTS/blob/master/notebooks/Benchmark.ipynb. We adapted the code from this ipython notebook to generate audio samples.

- download models
```
git clone -q https://github.com/mozilla/TTS.git
cd TTS && git checkout Tacotron2-iter-260K-824c091
pip install -q gdown lws librosa Unidecode==0.4.20 tensorboardX git+git://github.com/bootphon/phonemizer@master localimport
git clone -q https://github.com/erogol/WaveRNN.git
cd WaveRNN && git checkout 8a1c152 && pip install -q -r requirements.txt
# WaveRNN
mkdir -p wavernn_models tts_models
gdown -O wavernn_models/checkpoint_433000.pth.tar https://drive.google.com/uc?id=12GRFk5mcTDXqAdO5mR81E-DpTk8v2YS9
gdown -O wavernn_models/config.json https://drive.google.com/uc?id=1kiAGjq83wM3POG736GoyWOOcqwXhBulv
# TTS
gdown -O tts_models/checkpoint_261000.pth.tar https://drive.google.com/uc?id=1otOqpixEsHf7SbOZIcttv3O7pG0EadDx
gdown -O tts_models/config.json https://drive.google.com/uc?id=1IJaGo0BdMQjbnCcOL4fPOieOEWMOsXE-
```

### Generate TTS audio
- for original
```
mkdir tts-audio-original
python3 tts_generate.py [DATA_DST]/text/dev-other.txt tts-audio-original

```

- for 5 shuffled versions
```
python generate_shuffle_dev_other_tts.py [DATA_DST]/lists
for index in 0 1 2 3 4
do
  mkdir "tts-audio-$index"
  python3 tts_generate.py "tts_shuffled_$index.txt" "tts-audio-$index/tts-"
done
```
- convert to flac all files with
```
ffmpeg -i "$name.wav" -f flac "$name.flac"
```
- generate lists
```python
import os
import sys

# fix [DATA_DST]! path
for name, folder in zip(["tts_shuffled_0.txt", "tts_shuffled_1.txt", "tts_shuffled_2.txt", "tts_shuffled_3.txt", "tts_shuffled_4.txt", "[DATA_DST]/text/dev-other.txt"],
                        ["tts-audio-0", "tts-audio-1", "tts-audio-2", "tts-audio-3", "tts-audio-4", "tts-audio-original"]):
    with open(os.path.join(folder, "data.lst"), "w") as fout:
        with open(name, "r") as f:
            for index_f, line in enumerate(f):
                tr = line.strip()
                path = os.path.join(sys.argv[1], folder, "tts-" + str(index_f) + ".flac")
                duration = sox.file_info.duration(path) * 1000
                fout.write("{}\t{}\t{}\t{}\n".format(path, path, duration, tr))
```

### Compute WER for each list
Compute WER for each shuffled list `tts-audio-*/data.lst` (and then compute mean and std of WERs) and for original order `tts-audio-original/data.lst`
```
[...]/wav2letter/build/Test \
    --am [path/to/am/model.bin] \
    --tokensdir=[MODEL_DST]/am \
    --tokens=librispeech-train-all-unigram-10000.tokens \
    --lexicon=[MODEL_DST]/am/librispeech-train+dev-unigram-10000-nbest10.lexicon \
    --uselexicon=false \
    --datadir='' \
    --test=[DATA PART]/data.lst \
    --minloglevel=0 --logtostderr=1 \
    --maxtsz=1000000000 --maxisz=1000000000 --minisz=0 --mintsz=0 \
    --emission_dir=''
```

## Segmentation experiment

### Prepare lists
- Force alignment for dev-other set

The alignment ASG model and lexicon can be found in the [`lexicon_free recipe`](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/lexicon_free/librispeech#acoustic-model).
```
[...]/wav2letter/build/tools/Align \
    dev-other.lst.align  \
    --am=[PATH_TO_ALIGN_ASG_MODEL] \
    --test=dev-other.lst \
    --batchsize=1 \
    --datadir=[DATA_DST]/lists/ \
    --lexicon=[PATH_TO_ALIGN_MODEL_LEXICON]
```
- Filter the list
```
python3 filter_segmentations.py dev-other.lst.align [DATA_DST]/lists/dev-other.lst
```
- Prepare 5 different shuffle sets
```
mkdir seg_data_sil0.13s_tol0.04s_1
python3 shuffle_segments.py dev-other.lst.align.filtered_chunk_g1_ngrams_le6 `pwd`/seg_data_sil0.13s_tol0.04s_1
cat seg_data_sil0.13s_tol0.04s_1/dev-other.*.lst > seg_data_sil0.13s_tol0.04s_1/dev-other.lst
# repeat 4 times with suffix 2,3,4,5
```

### Compute WER for each list
Compute WER for each list `seg_data_sil0.13s_tol0.04s_*/dev-other.lst` and then compute mean and std of obtained WER. Compute for original dev-other with list `original.filtered_chunk_g1_ngrams_le6.lst`.
```
[...]/wav2letter/build/Test \
    --am=[path/to/am/model.bin] \
    --tokensdir=[MODEL_DST]/am \
    --tokens=librispeech-train-all-unigram-10000.tokens \
    --lexicon=[MODEL_DST]/am/librispeech-train+dev-unigram-10000-nbest10.lexicon \
    --uselexicon=false \
    --datadir='' \
    --test=[...] \
    --minloglevel=0 --logtostderr=1 \
    --maxtsz=1000000000 --maxisz=1000000000 --minisz=0 --mintsz=0 \
    --emission_dir=''
```
