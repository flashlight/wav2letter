# Steps to reproduce results on WSJ

## Reproduce acoustic model training
- Run data and auxiliary files (like lexicon, tokens set, etc.) preparation (set necessary paths instead of `[...]`: `data_dst` path to data to store, `model_dst` path to auxiliary path to store; if you are using `B`-version of WSJ then call with `--wsj1_type LDC94S13B`)
```
python3 prepare.py --wsj0 [...]/csr_1 --wsj1 [...]/csr_2_comp --wsj1_type LDC94S13A \
  --data_dst [...] --model_dst [...]  --sph2pipe [...]/sph2pipe_v2.5/sph2pipe
```
Besides data the auxiliary files for acoustic and language models training/evaluation will be generated:
```
cd $MODEL_DST
tree -L 2
.
├── am
│   ├── lexicon_si284+nov93dev.txt
│   ├── si284.lst.remap
│   └── tokens.lst
└── decoder
    ├── char_lm_data.nov93dev
    ├── char_lm_data.train
    ├── dict-remap.txt
    ├── lexicon.lst
    └── word_lm_data.nov93dev
```

- Fix the paths inside `*.cfg`
- We are run acoustic model training with `train.cfg` on **1 GPU**. Take `001_model_nov93dev.bin` snapshot for further decoder experiments.
```
[...]/wav2letter/build/Train train --flagsfile train.cfg --minloglevel=0 --logtostderr=1
```

## Reproduce language models training / evaluation
- prepare data for ConvLM training at first (use all words, no pruning)
```
source prepare_fairseq_data.sh [DATA_DST] [MODEL_DST] [FAIRSEQ PATH]
```
- train ngram models
```
./train_ngram_lms.sh [DATA_DST] [MODEL_DST] [KENLM PATH]/build/bin
```

- train fairseq models (all models are trained on a single machine with **8 GPUs** with fp16)
  - learning policy for word 14B model: `inverse_sqrt`, model overfits after 75k updates, we are taking this snapshots before perplexity on validation starts to grow.
```
mkdir -p [MODEL_DST]/decoder/convlm_models/word_14B
python3 [FAIRSEQ]/train.py [MODEL_DST]/decoder/fairseq_word_data \
--save-dir [MODEL_DST]/decoder/convlm_models/word_14B \
--task=language_modeling \
--arch=fconv_lm --fp16 --max-epoch=60 --optimizer=nag \
--lr=1 --lr-scheduler=inverse_sqrt --decoder-embed-dim=128 --clip-norm=0.1 \
--decoder-layers='[(512, 5)] + [(128, 1, 0), (128, 5, 0), (512, 1, 3)] * 3 + [(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3 + [(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 6 + [(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]' \
--dropout=0.3 --weight-decay=1e-07 \
--max-tokens=1024 --tokens-per-sample=1024 --sample-break-mode=none \
--criterion=adaptive_loss --adaptive-softmax-cutoff='10000,50000,100000' --seed=42 \
--log-format=json --log-interval=100 \
--save-interval-updates=10000 --keep-interval-updates=10 \
--ddp-backend="no_c10d" --distributed-world-size=8 > [MODEL_DST]/decoder/convlm_models/word_14B/train.log
```
  - learning policy for char 14B model: fixed lr, first 30 epochs use `lr=0.5`, then take best snapshot and run continue training with `lr=0.05` during another 30 epochs.
```
mkdir -p [MODEL_DST]/decoder/convlm_models/char_14B
python3 [FAIRSEQ]/train.py [MODEL_DST]/decoder/fairseq_char_data \
--save-dir [MODEL_DST]/decoder/convlm_models/char_14B --task=language_modeling \
--arch=fconv_lm --fp16 --max-epoch=60 --optimizer=nag \
--lr=0.5 --lr-scheduler="fixed" \
--decoder-embed-dim=128 --clip-norm=0.1 \
--decoder-layers='[(512, 5)] + [(128, 1, 0), (128, 5, 0), (512, 1, 3)] * 3 + [(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3 + [(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 6 + [(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]' \
--dropout=0.15 --weight-decay=1e-07 \
--max-tokens=512 --tokens-per-sample=512 --sample-break-mode=complete \
--criterion=cross_entropy --seed=42 \
--log-format=json --log-interval=100 \
--save-interval-updates=10000 --keep-interval-updates=10 \
--ddp-backend="no_c10d" --distributed-world-size=8 > [MODEL_DST]/decoder/convlm_models/char_14B/train.log
```
  - learning policy for char 20B model: for all epoch constant `lr=0.1` is used
```
mkdir -p [MODEL_DST]/decoder/convlm_models/char_20B
python3 [FAIRSEQ]/train.py [MODEL_DST]/decoder/fairseq_char_data \
--save-dir [MODEL_DST]/decoder/convlm_models/char_20B --task=language_modeling \
--arch=fconv_lm --fp16 --max-epoch=60 --optimizer=nag \
--lr=0.1 --lr-scheduler=fixed --decoder-embed-dim=256 --clip-norm=0.1 \
--decoder-layers='[(512, 5)] + [(128, 1, 0), (128, 5, 0), (256, 1, 3)] * 3 + [(256, 1, 0), (256, 5, 0), (512, 1, 3)] * 3 + [(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3 + [(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 9 + [(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]' \
--dropout=0.2 --weight-decay=1e-07 \
--max-tokens=512 --tokens-per-sample=512 --sample-break-mode=complete \
--criterion=cross_entropy --seed=42 \
--log-format=json --log-interval=100 \
--save-interval-updates=10000 --keep-interval-updates=10 \
--ddp-backend="no_c10d" --distributed-world-size=8 > [MODEL_DST]/decoder/convlm_models/char_20B/train.log
```
- compute upper and lower limit on word perplexity for trained LMs
```
# compute for ngram models
source eval_ngram_lms.sh [MODEL_DST]
source eval_convlm_lms.sh [DATA_DST] [MODEL_DST]
```

- convert ConvLM models into w2l format and generate vocabularies for decoding step
```
source convert_convlm.sh [MODEL_DST] [WAV2LETTER]/wav2letter
```

## Reproduce decoding
- Fix the paths inside `decoder*.cfg`
- Run decoding with `decoder*.cfg`
```
[...]/wav2letter/build/Decoder --flagsfile [...] --minloglevel=0 --logtostderr=1
```

## Pre-trained acoustic and language models
Below there is info about pre-trained acoustic and language models, which one can use, for example, to reproduce a decoding step:
- create dirs:
```
mkdir [MODEL_DST]/decoder/ngram_models [MODEL_DST]/decoder/convlm_models
```
- download an acoustic model into `$MODEL_DST/am`
- download language models (with their vocabularies for ConvLM) into `$MODEL_DST/decoder/ngram_models` for ngram models and into `$MODEL_DST/decoder/convlm_models` for ConvLM models
- download a lexicon (or generate it with the python script above) into `$MODEL_DST/decoder$`.

Here `am.arch`, generated `$MODEL_DST/am/tokens.lst` and `$MODEL_DST/decoder/lexicon.lst` files are the same as in the table.

### Acoustic Model
| File | Dataset | Dev Set | Architecture | Decoder lexicon | Tokens |
| - | - | - | - | - | - |
| [baseline_nov93dev](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/am/baseline_nov93dev.bin) | WSJ | nov93dev | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/am.arch) | [Decoder lexicon](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/lexicon.lst) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/tokens.lst) |

### Language Models

Convolutional language models (ConvLM) are trained with the [fairseq](https://github.com/pytorch/fairseq) toolkit. n-gram language models are trained with the [KenLM](https://github.com/kpu/kenlm) toolkit. The below language models are converted into a binary format compatible with the wav2letter++ decoder.

| Name |	Dataset | Type | Vocab | Fairseq model |
| - | - | - | - | - |
[lm_wsj_convlm_char_20B](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_convlm_char_20B.bin) | WSJ | ConvLM 20B | [LM Vocab](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_convlm_char_20B.vocab) | [Fairseq](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_convlm_char_20B.pt)
[lm_wsj_convlm_word_14B](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_convlm_word_14B.bin) | WSJ | ConvLM 14B | [LM Vocab](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_convlm_word_14B.vocab) | [Fairseq](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_convlm_word_14B.pt)
[lm_wsj_kenlm_char_15g_pruned](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_kenlm_char_15g_pruned.bin) | WSJ | 15-gram | - | -
[lm_wsj_kenlm_char_20g_pruned](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_kenlm_char_20g_pruned.bin) | WSJ | 20-gram | - | -
[lm_wsj_kenlm_word_4g](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/wsj/models/lm/lm_wsj_kenlm_word_4g.bin) | WSJ | 4-gram | - | -
