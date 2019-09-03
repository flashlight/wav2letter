# A Recipe for the Wall Street Journal (WSJ) corpus.

The WSJ corpus consists of about 80 hours of read sentences taken from the Wall Street Journal. The WSJ corpus can be purchased from the LDC:
- as [LDC93S6B](https://catalog.ldc.upenn.edu/LDC93S6B) (WSJ0) and [LDC94S13B](https://catalog.ldc.upenn.edu/LDC94S13B) (WSJ1)
- or the complete version of datasets: [LDC93S6A](https://catalog.ldc.upenn.edu/LDC93S6A) (WSJ0) and [LDC94S13A](https://catalog.ldc.upenn.edu/LDC94S13A) (WSJ1)

In these experiments, we use three subsets following the [Kaldi WSJ recipe](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/):

- train: 37416 utterances, referred to as si284 in Kaldi
- dev: 503 utterances, referred to as nov93dev in Kaldi
- test: 333 utterances, referred to as nov92 in Kaldi

## Wav2Letter models
| NOV93DEV WER % / LER % | NOV92 WER % / LER % | MODEL                                                                                                      | PAPER                                                                                                     |
|-------------------------|-------------------------|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| 9.8 / 7.2 | 5.6 / - | [conv_glu](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/conv_glu/wsj) | [Wav2Letter: an End-to-End ConvNet-based Speech Recognition System](https://arxiv.org/pdf/1609.03193.pdf), [Letter-Based Speech Recognition with Gated ConvNets](https://arxiv.org/pdf/1712.09444.pdf) |

## Prerequisites
Here and later we assume that complete version of data is downloaded, but the same steps can be used for `B`-version of WSJ.

- download the data from the LDC [WSJ0](https://catalog.ldc.upenn.edu/LDC93S6A) and [WSJ1](https://catalog.ldc.upenn.edu/LDC94S13A). You will have `csr_1_LDC93S6A.tar` and `csr_2_comp_LDC94S13A.tar`.

- unpack files
```
tar -xf csr_1_LDC93S6A.tar
tar -xf csr_2_comp_LDC94S13A.tar
```

- install `sph2pipe` see https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools:
```  
wget https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz
tar -xzf sph2pipe_v2.5.tar.gz && cd sph2pipe_v2.5
gcc -o sph2pipe *.c -lm
```

## Preparation of audio and text data

To prepare the audio and text data for training/evaluation run (set necessary paths instead of `[...]`)
(if you are using `B`-version of WSJ then call with `--wsj1_type LDC94S13B`)
```
python3 prepare.py --wsj0 [...]/csr_1 --wsj1 [...]/csr_2_comp --sph2pipe [...]/sph2pipe_v2.5/sph2pipe --dst [...] --wsj1_type LDC94S13A
```

The following structure will be generated
```
tree -L 2
.
├── audio
│   ├── nov92
│   ├── nov92_5k
│   ├── nov93
│   ├── nov93_5k
│   ├── nov93dev
│   ├── nov93dev_5k
│   ├── si284
│   └── si84
├── csr_1
│   ├── 11-10.1
│   ├── ...
│   └── readme.txt
├── csr_1_LDC93S6A.tar
├── csr_2_comp
│   ├── 13-10.1
│   ├── ...
├── csr_2_comp_LDC94S13A.tar
├── lists
│   ├── nov92_5k.lst
│   ├── nov92.lst
│   ├── nov93_5k.lst
│   ├── nov93dev_5k.lst
│   ├── nov93dev.lst
│   ├── nov93.lst
│   ├── si284.lst
│   └── si84.lst
└── text
    ├── lm.txt
    ├── nov92_5k.txt
    ├── nov92.txt
    ├── nov93_5k.txt
    ├── nov93dev_5k.txt
    ├── nov93dev.txt
    ├── nov93.txt
    ├── si284.txt
    └── si84.txt
```
