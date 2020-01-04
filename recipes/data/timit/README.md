# A Recipe for the TIMIT corpus.

The TIMIT corpus can be purchased from the LDC [LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1).

The data is mapped from 61 to 39 phonemes for training and testing. The map used here is taken from the [Kaldi TIMIT recipe](https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/phones.60-48-39.map).

## Prerequisites
- download the data from the LDC [LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1). You will have `timit_LDC93S1.tar`.
- unpack file
```
tar -xf timit_LDC93S1.tar
```
- install `sph2pipe` see https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools:
```  
wget https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz
tar -xzf sph2pipe_v2.5.tar.gz && cd sph2pipe_v2.5
gcc -o sph2pipe *.c -lm
```

## Preparation of audio and text data

To prepare the audio and text data for training/evaluation run (set necessary paths instead of `[...]`)
```
python3 prepare.py --src [...]/timit --dst [...] --sph2pipe [...]/sph2pipe_v2.5/sph2pipe
```

The following structure will be generated
```
tree -L 2
.
├── audio
│   ├── test
│   ├── train
│   └── valid
├── lists
│   ├── test.lst
│   ├── test.phn.lst
│   ├── train.lst
│   ├── train.phn.lst
│   ├── valid.lst
│   └── valid.phn.lst
├── text
│   ├── test.phn.txt
│   ├── test.txt
│   ├── train.phn.txt
│   ├── train.txt
│   ├── valid.phn.txt
│   └── valid.txt
├── timit
│   ├── CONVERT
│   ├── README.DOC
│   ├── SPHERE
│   └── TIMIT
└── timit_LDC93S1.tar
```
