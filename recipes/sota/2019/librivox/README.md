# Acoustic models training on Librivox dataset

- Fix the paths inside `train*.cfg`

### Resnet CTC training
The model is trained with total batch size 256 for approximately 55 epochs. Optimizer is SGD with momentum.
```
[...]/wav2letter/build/Train train --flagsfile train_am_resnet_ctc.cfg --minloglevel=0 --logtostderr=1
# after 40 epochs
[...]/wav2letter/build/Train continue [path/to/am/model.bin] --lr=0.03 --lrcrit=0.03 --minloglevel=0 --logtostderr=1
```

### Resnet S2S training
The model is trained with total batch size 256 for approximately 50 epochs. Optimizer is SGD with momentum.
```
[...]/wav2letter/build/Train train --flagsfile train_am_resnet_s2s.cfg --minloglevel=0 --logtostderr=1
# after 36 epochs
[...]/wav2letter/build/Train continue [path/to/am/model.bin] --lr=0.03 --lrcrit=0.03 --minloglevel=0 --logtostderr=1
```

### TDS CTC training
The model is trained with total batch size 256 for approximately 30 epochs. Optimizer is SGD with momentum.
```
[...]/wav2letter/build/Train train --flagsfile train_am_tds_ctc.cfg --minloglevel=0 --logtostderr=1
# after 26 epochs
[...]/wav2letter/build/Train continue [path/to/am/model.bin] --lr=0.15 --minloglevel=0 --logtostderr=1
```

### TDS Seq2Seq training
The model is trained with total batch size 256 for approximately 50 epochs. Optimizer is SGD with momentum.
```
[...]/wav2letter/build/Train train --flagsfile train_am_tds_s2s.cfg --minloglevel=0 --logtostderr=1
# after 37 epochs
[...]/wav2letter/build/Train continue [path/to/am/model.bin] --lr=0.025 --lrcrit=0.025 --minloglevel=0 --logtostderr=1
```

### Transformer CTC training
The model is trained with total batch size 320 for approximatively 23 epochs with Adagrad.
There is a warmup stage: SpecAugment is activated only after warmup, and the learning rate is warmed up (linearly increased) over the first 30000 updates to 0.02. It is then divided by 2 at epoch 5, and then every 4 epochs.
```
[...]/wav2letter/build/Train train --flagsfile train_am_transformer_ctc.cfg --minloglevel=0 --logtostderr=1
# last 12 epochs are done with
[...]/wav2letter/build/Train continue [path/to/am/model.bin] --maxisz=33000 --minloglevel=0 --logtostderr=1
```

### Transformer Seq2Seq training
The model is trained with total batch size 320 for approximatively 22 epochs with Adagrad and SGD finetuning.
There is a warmup stage: SpecAugment is activated only after warmup, and the learning rate is warmed up (linearly increased) over the first 40000 updates to 0.02. It is then divided by 2 at epoch 4, and then every 3 epochs.
```
[...]/wav2letter/build/Train train --flagsfile train_am_transformer_s2s.cfg --minloglevel=0 --logtostderr=1
# after 9 epochs we continue with
[...]/wav2letter/build/Train continue [path/to/am/model.bin] --maxisz=33000 --minloglevel=0 --logtostderr=1
# after 15 epochs we finetune with SGD
[...]/wav2letter/build/Train continue [path/to/am/model.bin] --minloglevel=0 --logtostderr=1 --lr=0.01 --lrcrit=0.01 --lr_decay=5 --lr_decay_step=5 --netoptim=sgd --critoptim=sgd --warmup=0 --pretrainWindow=0 --enable_distributed
```


## Greedy search (To get WER without beam-search decoding, = Viterbi WER)
- Run test (put the paths instead of `[path/to/am/model.bin]`, `[MODEL_DST]`, `[DATA_DST]`), for example for test-other
```
[...]/wav2letter/build/Test \
    --am=[path/to/am/model.bin] \
    --tokensdir=[MODEL_DST]/am \
    --tokens=librispeech-train-all-unigram-10000.tokens \
    --lexicon=[MODEL_DST]/am/librispeech-train+dev-unigram-10000-nbest10.lexicon \
    --uselexicon=false \
    --datadir=[DATA_DST]/lists \
    --test=test-other.lst \
    --minloglevel=0 --logtostderr=1 \
    --maxtsz=1000000000 --maxisz=1000000000 --minisz=0 --mintsz=0 \
    --emission_dir=''

```

## Beam-search decoding
- Fix the paths inside `decode*.cfg`
- Run decoding with `decode*.cfg`
```
[...]/wav2letter/build/Decoder --flagsfile=path/to/necessary/decode/config --minloglevel=0 --logtostderr=1 --maxtsz=1000000000 --maxisz=1000000000 --minisz=0 --mintsz=0 --emission_dir=''
```
