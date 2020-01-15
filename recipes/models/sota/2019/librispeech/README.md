# Acoustic models training and beam-search decoding on Librispeech

## Acoustic models training
- Fix the paths inside `train*.cfg`

### Resnet CTC training
The model is trained with total batch size 128 for approximately 450 epochs.
```
[...]/wav2letter/build/Train train --flagsfile train_am_resnet_ctc.cfg --minloglevel=0 --logtostderr=1
```

### TDS CTC training
The model is trained with total batch size 256 for approximately 600 epochs.
```
[...]/wav2letter/build/Train train --flagsfile train_am_tds_ctc.cfg --minloglevel=0 --logtostderr=1
```

### TDS Seq2Seq training
The model is trained with total batch size 128 for approximately 1400 epochs.
```
[...]/wav2letter/build/Train train --flagsfile train_am_tds_s2s.cfg --minloglevel=0 --logtostderr=1
```

### Transformer CTC training
The model is trained with total batch size 128 for approximatively 500 epochs with Adagrad.
There is a warmup stage: SpecAugment is activated only after warmup, and the learning rate is warmed up (linearly increased) over the first 64000 updates to 0.03. It is then divided by 2 at epoch 160, and then every 70 epochs.
```
[...]/wav2letter/build/Train train --flagsfile train_am_transformer_ctc.cfg --minloglevel=0 --logtostderr=1
```

### Transformer Seq2Seq training
The model is trained with total batch size 256 for approximatively 550 epochs with Adagrad.
There is a warmup stage: SpecAugment is activated only after warmup, and the learning rate is warmed up (linearly increased) over the first 64000 updates to 0.03. It is then divided by 2 at epoch 200, and then every 40 epochs.
```
[...]/wav2letter/build/Train train --flagsfile train_am_transformer_s2s.cfg --minloglevel=0 --logtostderr=1
```

## Greedy search
- Run test (put the paths instead of `[path/to/am/model.bin]`, `[MODEL_DST]`, `[DATA_DST]`), for example for test-other
```
[...]/wav2letter/build/Test \
    --am [path/to/am/model.bin] \
    --tokensdir=[MODEL_DST]/am \
    --tokens=librispeech-train-all-unigram-10000.tokens \
    --lexicon=[MODEL_DST]/am/librispeech-train+dev-unigram-10000-nbest10.lexicon \
    --uselexicon=false \
    --datadir=[DATA_DST]/lists \
    --test=test-other.lst \
    --minloglevel=0 --logtostderr=1
```

## Beam-search decoding
- Fix the paths inside `decode*.cfg`
- Run decoding with `decode*.cfg`
```
[...]/wav2letter/build/Decoder --flagsfile path/to/necessary/decode/config --minloglevel=0 --logtostderr=1
```
