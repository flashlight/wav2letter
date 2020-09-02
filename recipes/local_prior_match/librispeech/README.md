# Steps to reproduce results on Librispeech

Run data and auxiliary files (like lexicon, tokens set, etc.) preparation (set necessary paths instead of `[...]`: `data_dst` path to data to store, `model_dst` path to auxiliary path to store).
```
pip install sentencepiece==0.1.82
python3 prepare.py --data_dst [...] --model_dst [...]
```
Besides data the auxiliary files for acoustic and language models training/evaluation will be generated:
```
cd $MODEL_DST
tree -L 2
.
├── am
│   ├── librispeech-paired-train+dev-unigram-5000-nbest10.lexicon
│   └── librispeech-paired-train-unigram-5000.tokens
└── lm
└── lpm_data
    ├── train-clean-360-dummy.lst
    └── train-other-500-dummy.lst
```

Build wav2letter++ with `-DW2L_BUILD_RECIPES=ON`. In addition to the top-level binaries such as `Train`, binaries specific to local prior matching, `decode_len_lpm` and `Train_lpm_oss`, will be built under `[...]/recipes/models/local_prior_match`.

Training consists of the following steps:
1. Train an initial model and a proposal model using the main wav2letter training binary (can be done in parallel)
  - Fix the paths inside `train_init.cfg` and `train_proposal.cfg`
  - Train an initial model with
  `[...]/Train train --flagsfile=train_init.cfg`
  - Train a proposal model with
  `[...]/Train train --flagsfile=trian_proposal.cfg`
  - Note that the parameters and settings in `train_init.cfg` and `train_proposal.cfg` are for running experiments on a single GPU.
2. Use the proposal model to decode on unpaired data to generate reference length. (Try split the unpaired data into smaller subsets and parallelize the process if the dataset is huge.)
  ```
  # use the best model from the last run
  [...]/decode_len_lpm [rundir]/lpm_proposal/[xxx]_model_dev-clean.bin \
      [model_dst]/lpm_data/train-clean-360-dummy.lst \
      [model_dst]/lpm_data/train-clean-360-viterbi.out

  [...]/decode_len_lpm [rundir]/lpm_proposal/[xxx]_model_dev-other.bin \
      [model_dst]/lpm_data/train-other-500-dummy.lst \
      [model_dst]/lpm_data/train-other-500-viterbi.out
  ```
3. Prepare the unpaired data
```
python3 prepare_unpaired.py --data_dst [...] --model_dst [...]
```
In addition to the files generated from `prepare.py`, the following files will be generated under `$MODEL_DST` for LPM training:
```
cd $MODEL_DST
.
├── am
│   └── librispeech-paired-train-unpaired-viterbi+dev-unigram-5000-nbest10.lexicon
└── lm
└── lpm_data
    ├── train-clean-360-lpm.lst
    └── train-other-500-lpm.lst
```
4. Train an LPM model
  - Download the [LM](https://dl.fbaipublicfiles.com/wav2letter/lpm/librispeech/models/lm/lpm_librispeech_lm.bin) and [dictionary](https://dl.fbaipublicfiles.com/wav2letter/lpm/librispeech/models/lm/lm_dict.txt) to `[model_dst]/lm`
  - Fix the paths and the proposal model name (id from the last run) inside `train_lpm.cfg`
  - Train an LPM model with
  `[...]/Train_lpm_oss fork [rundir]/lpm_init/[xxx]_model_last.bin --flagsfile=train_lpm.cfg`
  - Note that the parameters and settings in `train_lpm.cfg` are for running experiments on a single node with **8 GPUs** (`--enable_distributed=true`). Distributed jobs can be launched using [Open MPI](https://www.open-mpi.org/).
