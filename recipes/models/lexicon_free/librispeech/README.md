# Steps to reproduce results on Librispeech

## Reproduce acoustic model training
- Prepare lexicon and tokens for acoustic model training (set necessary paths instead of `[...]`: `data_dst` path to data to store, `model_dst` path to auxiliary path to store, `kenlm` path)
```
python3 prepare.py --data_dst [...] --model_dst [...] --kenlm [...]
```
The following files will be generated:
```
cd $MODEL_DST
tree -L 2
.
├── am
│   ├── lexicon_train+dev.lst
│   └── tokens.lst
└── decoder
    ├── 4-gram.arpa
    ├── 4-gram.bin
    ├── char_lm_data.dev-clean
    ├── char_lm_data.dev-other
    ├── char_lm_data.train
    ├── lexicon.lst
    ├── test-clean.lst.inv
    ├── test-clean.lst.oov
    ├── test-other.lst.inv
    └── test-other.lst.oov

```
- Fix the paths inside `train.cfg`
- We are running acoustic model training with `train.cfg` on a single node with **8 GPUs** (distributed jobs can be launched using [Open MPI](https://www.open-mpi.org/). During training we are decreasing the learning rate:
```
[...]/wav2letter/build/Train train --flagsfile train.cfg --minloglevel=0 --logtostderr=1
[...]/wav2letter/build/Train continue [PATH/TO/MODEL/DIR] --linseg=0 --enable_distributed --lr=0.1 --lrcrit=0.001 --maxgradnorm=0.25 --iter=7 --minloglevel=0 --logtostderr=1
[...]/wav2letter/build/Train continue [PATH/TO/MODEL/DIR] --linseg=0 --enable_distributed --lr=0.1 --lrcrit=0.001 --maxgradnorm=0.25 --stepsize 4 --gamma 0.9 --iter=70 --minloglevel=0 --logtostderr=1
```
Take either `003_model_librispeech_dev-clean.bin` or `003_model_librispeech_dev-other.bin`. We are using `003_model_librispeech_dev-clean.bin` snapshot for further decoder experiments.

## Reproduce language models training / evaluation

- prepare data for ConvLM training at first (use its dictionary to define vocabulary file for 4gram word LM training)
```
source prepare_fairseq_data.sh [DATA_DST] [MODEL_DST] [FAIRSEQ PATH]
```
- train ngram models
```
./train_ngram_lms.sh [DATA_DST] [MODEL_DST] [KENLM PATH]/build/bin
```
- train fairseq models (all models are trained on a single machine with **8 GPUs** with fp16)
  - learning policy for word 14B model: we trained 11 epochs with `lr=0.5`, then till 30th epoch with `lr=0.05`, and then till 48th epoch with `lr=0.005`.
```
mkdir -p [MODEL_DST]/decoder/convlm_models/word_14B
python3 [FAIRSEQ]/train.py [MODEL_DST]/decoder/fairseq_word_data \
--save-dir [MODEL_DST]/decoder/convlm_models/word_14B \
--task=language_modeling \
--arch=fconv_lm --fp16 --max-epoch=48 --optimizer=nag \
--lr=0.5 --lr-scheduler=fixed --decoder-embed-dim=128 --clip-norm=0.1 \
--decoder-layers='[(512, 5)] + [(128, 1, 0), (128, 5, 0), (512, 1, 3)] * 3 + [(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3 + [(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 6 + [(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]' \
--dropout=0.1 --weight-decay=1e-07 \
--max-tokens=1024 --tokens-per-sample=1024 --sample-break-mode=none \
--criterion=adaptive_loss --adaptive-softmax-cutoff='10000,50000,200000' --seed=42 \
--log-format=json --log-interval=100 \
--save-interval-updates=10000 --keep-interval-updates=10 \
--ddp-backend="no_c10d" --distributed-world-size=8 > [MODEL_DST]/decoder/convlm_models/word_14B/train.log
```
  - learning policy for char 14B model: reducing on plateau.
```
mkdir -p [MODEL_DST]/decoder/convlm_models/char_14B
python3 [FAIRSEQ]/train.py [MODEL_DST]/decoder/fairseq_char_data \
--save-dir [MODEL_DST]/decoder/convlm_models/char_14B --task=language_modeling \
--arch=fconv_lm --fp16 --max-epoch=48 --optimizer=nag \
--lr=0.5 --lr-scheduler="reduce_lr_on_plateau" --lr-shrink=0.7 \
--decoder-embed-dim=128 --clip-norm=0.1 \
--decoder-layers='[(512, 5)] + [(128, 1, 0), (128, 5, 0), (512, 1, 3)] * 3 + [(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3 + [(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 6 + [(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]' \
--dropout=0.1 --weight-decay=1e-07 \
--max-tokens=512 --tokens-per-sample=512 --sample-break-mode=complete \
--criterion=cross_entropy --seed=42 \
--log-format=json --log-interval=100 \
--save-interval-updates=10000 --keep-interval-updates=10 \
--ddp-backend="no_c10d" --distributed-world-size=8 > [MODEL_DST]/decoder/convlm_models/char_14B/train.log
```
  - learning policy for char 20B model: for all epoch constant `lr=0.5` is used
```
mkdir -p [MODEL_DST]/decoder/convlm_models/char_20B
python3 [FAIRSEQ]/train.py [MODEL_DST]/decoder/fairseq_char_data \
--save-dir [MODEL_DST]/decoder/convlm_models/char_20B --task=language_modeling \
--arch=fconv_lm --fp16 --max-epoch=16 --optimizer=nag \
--lr=0.5 --lr-scheduler=fixed --decoder-embed-dim=256 --clip-norm=0.1 \
--decoder-layers='[(512, 5)] + [(128, 1, 0), (128, 5, 0), (256, 1, 3)] * 3 + [(256, 1, 0), (256, 5, 0), (512, 1, 3)] * 3 + [(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3 + [(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 9 + [(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]' \
--dropout=0.1 --weight-decay=1e-07 \
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
- to compute WER(CER) on sentences with at least one out-of-vocabulary word and
on sentences with only in-vocabulary words separately use lists `$MODEL_DST/decoder/test-*.lst.oov` and `$MODEL_DST/decoder/test-*.lst.inv` correspondently in the `decoder*.cfg`


# Pre-trained acoustic and language models
Below there is info about pre-trained acoustic and language models, which one can use, for example, to reproduce a decoding step:
- create dirs:
```
mkdir [MODEL_DST]/decoder/ngram_models [MODEL_DST]/decoder/convlm_models
```
- download an acoustic model into `$MODEL_DST/am`
- download language models (with their vocabularies for ConvLM) into `$MODEL_DST/decoder/ngram_models` for ngram models and into `$MODEL_DST/decoder/convlm_models` for ConvLM models
- download a lexicon (or generate it with the python script above) into `$MODEL_DST/decoder$`.

Here `am.arch`, generated `$MODEL_DST/am/tokens.lst` and `$MODEL_DST/decoder/lexicon.lst` files are the same as in the table.

## Acoustic Model
| File | Dataset | Dev Set | Architecture | Decoder Lexicon | Tokens |
| - | - | - | - | - | - |
| [baseline_dev-clean+other](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/am/baseline_dev-clean%2Bother.bin) | LibriSpeech | dev-clean+dev-other | [Archfile](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/am.arch) | [Decoder lexicon](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/lexicon.lst) | [Tokens](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/tokens.lst) |

## Language Models

Convolutional language models (ConvLM) are trained with the [fairseq](https://github.com/pytorch/fairseq) toolkit. n-gram language models are trained with the [KenLM](https://github.com/kpu/kenlm) (for ngram language models training) toolkit. The below language models are converted into a binary format compatible with the wav2letter++ decoder.

| Name |	Dataset | Type | Vocab | Fairseq model |
| - | - | - | - | - |
[lm_librispeech_convlm_char_20B](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_char_20B.bin) | LibriSpeech | ConvLM 20B | [LM Vocab](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_char_20B.vocab) | [Fairseq](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_char_20B.pt)
[lm_librispeech_convlm_word_14B](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_word_14B.bin) | LibriSpeech | ConvLM 14B | [LM Vocab](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_word_14B.vocab) | [Fairseq](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_convlm_word_14B.pt)
[lm_librispeech_kenlm_char_15g_pruned](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_kenlm_char_15g_pruned.bin) | LibriSpeech | 15-gram | - | -
[lm_librispeech_kenlm_char_20g_pruned](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_kenlm_char_20g_pruned.bin) | LibriSpeech | 20-gram | - | -
[lm_librispeech_kenlm_word_4g_200kvocab](https://dl.fbaipublicfiles.com/wav2letter/lexicon_free/librispeech/models/lm/lm_librispeech_kenlm_word_4g_200kvocab.bin) | LibriSpeech | 4-gram | - | -
