# Language model training

Word based 4-gram and word-based GCNN models are used from the paper [Who Needs Words? Lexicon-Free Speech Recognition](https://arxiv.org/abs/1904.04479), detail one can find in the [recipe](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/lexicon_free/librispeech)

## Dependencies
- [fairseq](https://github.com/pytorch/fairseq), commit `d80ad54`
- [KenLM](https://github.com/kpu/kenlm), commit `e47088d`

## Data Preparation
```
# set necessary paths before running
python3 prepare_wp_data.py --data_src [DATA_DST] --model_src [MODEL_DST]
mkdir -p "$MODEL_DST/decoder/fairseq_wp_10k_data/"

python "$FAIRSEQ/preprocess.py" --only-source \
--trainpref "$MODEL_DST/decoder/lm_wp_10k.train" \
--validpref "$MODEL_DST/decoder/lm_wp_10k.dev-clean" \
--testpref "$MODEL_DST/decoder/lm_wp_10k.dev-other" \
--destdir "$MODEL_DST/decoder/fairseq_wp_10k_data/" \
--workers 16
```

## Training
- 10k word-pieces (wp) 6-gram model trained with KenLM
```
export KENLM=path/to/kenlm/build/bin
export MODEL_DST=[...]
export MODEL_ARPA="$MODEL_DST/decoder/lm_librispeech_kenlm_wp_10k_6gram_pruning_000012.arpa"
export MODEL_BIN="$MODEL_DST/decoder/lm_librispeech_kenlm_wp_10k_6gram_pruning_000012.bin"
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback --prune 0 0 0 0 1 2 -o 6 trie < "$MODEL_DST/decoder/lm_wp_10k.train" > "$MODEL_ARPA"
"$KENLM/build_binary" trie "$MODEL_ARPA" "$MODEL_BIN"
"$KENLM/query" "$MODEL_BIN" <"$MODEL_DST/decoder/lm_wp_10k.dev-clean" > "$MODEL_BIN".dev-clean
"$KENLM/query" lm_librispeech_kenlm_wp_10k_6gram_pruning_000012.bin <"$MODEL_DST/decoder/lm_wp_10k.dev-other" > "$MODEL_BIN".dev-other
```

- 10k word-pieces (wp) GCNN training on **8 GPUs** with the following learning rate policy: we trained 6 epochs with `lr=1.0`, then till 15th epoch with `lr=0.2`, then till 20th epoch with `lr=0.04`, and then 2 more epochs with `lr=0.008`.  
```
mkdir -p lm_librispeech_wp_10k_gcnn_14B
python3 [FAIRSEQ]/train.py [MODEL_DST]/decoder/fairseq_wp_10k_data \
--save-dir lm_librispeech_wp_10k_gcnn_14B \
--task=language_modeling \
--arch=fconv_lm --fp16 --max-epoch=21 --optimizer=nag \
--lr= --lr-scheduler=fixed --decoder-embed-dim=1024 --clip-norm=0.1 \
--decoder-layers='[(512, 5)] + [(128, 1, 0), (128, 5, 0), (512, 1, 3)] * 3 + [(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3 + [(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 6 + [(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]' \
--dropout=0.1 --weight-decay=1e-07 \
--max-tokens=1024 --tokens-per-sample=128 --sample-break-mode=complete \
--criterion=cross_entropy --seed=42 \
--log-format=json --log-interval=100 \
--save-interval-updates=10000 --keep-interval-update=10 \
--ddp-backend="no_c10d" --distributed-world-size=8 > lm_librispeech_wp_10k_gcnn_14B/train.log
```

## Results (Word-level perplexity, for wp model it is an upper limit)

| Language Model | params | dev-clean | dev-other |
|:-:|:-:|:-:|:-:|
| word 4-gram | - | 148 | 136.6 |
| word-pieces 6-gram | - | 145.4 | 133.7 |
| word-pieces GCNN | 188M | 61.7 | 61.9 |
| word-based GCNN | 319M | 57 | 57.9 |
| Transformer | 429M | 48.2 | 50.2 |
