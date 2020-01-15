# Acoustic models training and beam-search decoding on Librispeech and Librilight (with generated pseudo labels)

## Acostic models training
- Fix the paths inside `train*.cfg`

### Resnet CTC training
The model is trained with total batch size 128 for 25 epochs.
```
[...]/wav2letter/build/Train train --flagsfile train_am_resnet_ctc.cfg --minloglevel=0 --logtostderr=1
```

### TDS CTC training
The model is trained with total batch size 128 for 9 epochs. There
There is a warmup stage: SpecAugment is activated only after warmup, and the learning rate is warmed up (linearly increased) over the first 64000 updates to 0.4.
```
[...]/wav2letter/build/Train train --flagsfile train_am_tds_ctc.cfg --minloglevel=0 --logtostderr=1
```

### TDS Seq2Seq training
The model is trained with total batch size 256. AT first, we train on Librispeech only for 15 epochs, and then we train on Librispeech + LibriVox for approximately 60 epochs.
```
[...]/wav2letter/build/Train train --flagsfile train_am_tds_s2s.cfg --minloglevel=0 --logtostderr=1
```

### Transformer CTC training
The model is trained with total batch size 128 for 6 epochs with Adagrad.
There is a warmup stage: SpecAugment is activated only after warmup, and the learning rate is warmed up (linearly increased) over the first 64000 updates to 0.02. It is then divided by 2 at epoch 3, and then every 1 epoch.
```
[...]/wav2letter/build/Train train --flagsfile train_am_transformer_ctc.cfg --minloglevel=0 --logtostderr=1
```

### Transformer Seq2Seq training
The model is trained with total batch size 256 for 16 epochs with Adagrad.
There is a warmup stage: SpecAugment is activated only after warmup, and the learning rate is warmed up (linearly increased) over the first 64000 updates to 0.03. It is then divided by 2 at epoch 3. At epoch 8 learning rate is set to `--lr=0.0075`, and at epoch 14 - `--lr=0.005`.
```
[...]/wav2letter/build/Train train --flagsfile train_am_transformer_s2s.cfg --minloglevel=0 --logtostderr=1
[...]/wav2letter/build/Train continue [path/to/model]/am_transformer_s2s_librivox --flagsfile train_am_transformer_s2s.cfg --minloglevel=0 --logtostderr=1 --lr=0.0075 --lrcrit=0.0075
[...]/wav2letter/build/Train continue [path/to/model]/am_transformer_s2s_librivox --flagsfile train_am_transformer_s2s.cfg --minloglevel=0 --logtostderr=1 --lr=0.005 --lrcrit=0.005
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
