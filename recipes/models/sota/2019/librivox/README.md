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

### Transformer CTC training
The model is trained with total batch size 128 for 6 epochs with Adagrad.
There is a warmup stage: SpecAugment is activated only after warmup, and the learning rate is warmed up (linearly increased) over the first 64000 updates to 0.02. It is then divided by 2 at epoch 3, and then every 1 epoch.
```
[...]/wav2letter/build/Train train --flagsfile train_am_transformer_ctc.cfg --minloglevel=0 --logtostderr=1
```

## Beam-search decoding
- Fix the paths inside `decode*.cfg`
- Run decoding with `decode*.cfg`
```
[...]/wav2letter/build/Decoder --flagsfile path/to/necessary/decode/config --minloglevel=0 --logtostderr=1
```
