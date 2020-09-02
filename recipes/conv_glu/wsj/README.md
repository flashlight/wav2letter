# Steps to reproduce results on WSJ

Run data and auxiliary files (like lexicon, tokens set, etc.) preparation (set necessary paths instead of `[...]`: `data_dst` path to data to store, `model_dst` path to auxiliary path to store, `kenlm` path to prepare language model for decoding; if you are using `B`-version of WSJ then call with `--wsj1_type LDC94S13B`)
```
python3 prepare.py --wsj0 [...]/csr_1 --wsj1 [...]/csr_2_comp --wsj1_type LDC94S13A \
  --data_dst [...] --model_dst [...]  --sph2pipe [...]/sph2pipe_v2.5/sph2pipe --kenlm [...]
```
Besides data the auxiliary files for acoustic and language models training/evaluation will be generated:
```
cd $MODEL_DST
tree -L 2
.
├── am
│   ├── lexicon_si284+nov93dev.txt
│   └── tokens.txt
└── decoder
    ├── lexicon.txt # it is combination of words from si284 and lm text data
    ├── lm-4g.arpa
    └── lm-4g.bin
```

To run training/decoding:
- Fix the paths inside `*.cfg`
- Run training with `train.cfg`
- Run decoding with `decode*.cfg`
