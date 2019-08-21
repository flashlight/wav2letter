# Steps to reproduce results on Librispeech

Run data and auxiliary files (like lexicon, tokens set, etc.) preparation (set necessary paths instead of `[...]`: `data_dst` path to data to store, `model_dst` path to auxiliary path to store, `kenlm` path to prepare language model for decoding)
```
python3 prepare.py --data_dst [...] --model_dst [...] --kenlm [...]
```
Besides data the auxiliary files for acoustic and language models training/evaluation will be generated:
```
cd $MODEL_DST
tree -L 2
.
├── am
│   ├── lexicon_train+dev.txt
│   └── tokens.txt
└── decoder
    ├── 4-gram.arpa
    ├── 4-gram.bin
    └── lexicon.txt
```

To run training/decoding:
- Fix the paths inside `*.cfg`
- Run training with `train.cfg`
- Run decoding with `decode*.cfg`
