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
mkdir -p "$MODEL_DST/decoder/fairseq_word_data/"

# for wp LM
python "$FAIRSEQ/preprocess.py" --only-source \
--trainpref "$MODEL_DST/decoder/lm_wp_10k.train" \
--validpref "$MODEL_DST/decoder/lm_wp_10k.dev-clean" \
--testpref "$MODEL_DST/decoder/lm_wp_10k.dev-other" \
--destdir "$MODEL_DST/decoder/fairseq_wp_10k_data/" \
--workers 16

# for word transformer LM
python "$FAIRSEQ/preprocess.py" --only-source \
--trainpref "$DATA_DST/text/librispeech-lm-norm.txt.lower.shuffle" \
--validpref "$DATA_DST/text/dev-clean.txt" \
--testpref "$DATA_DST/text/dev-other.txt" \
--destdir "$MODEL_DST/decoder/fairseq_word_data/" \
--thresholdsrc 10 \
--workers 16

```

## Training
- 10k word-pieces (wp) 6-gram model is trained with KenLM
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

- 10k word-pieces (wp) GCNN is trained on **8 GPUs** with the following learning rate policy: we trained 6 epochs with `lr=1.0`, then till 15th epoch with `lr=0.2`, then till 20th epoch with `lr=0.04`, and then 2 more epochs with `lr=0.008`.
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

# eval
cd [FAIRSEQ] && python eval_lm.py [MODEL_DST]/decoder/fairseq_wp_10k_data --gen-subset valid --sample-break-mode eos --seed 42 --tokens-per-sample 1024 --max-tokens=1024 --path lm_librispeech_wp_10k_gcnn_14B/checkpoint_best.pt
```

- word Transformer LM is trained on **128 GPUs** with the following learning rate policy: 48 epochs we train with `sqrt_inv` schedule with `lr=1` (16000 updates of warmup is done at the beginning)
```
mkdir -p lm_librispeech_word_transformer
python3 [FAIRSEQ]/train.py [MODEL_DST]/decoder/fairseq_word_data \
--save-dir lm_librispeech_word_transformer \
--task=language_modeling \
--arch=transformer_lm_gbw --fp16 --optimizer=nag --clip-norm=0.1 \
--criterion='adaptive_loss' --adaptive-softmax-cutoff='60000,160000' \
--adaptive-input=True --adaptive-input-cutoff='60000,160000' \
--adaptive-input-factor=4 --adaptive-softmax-factor=4 \
--adaptive_softmax_dropout=0 --tie-adaptive-weights=True --tie_adaptive_proj=False \
--decoder-layers=20 --decoder-attention-heads=16 \
--decoder-ffn-embed-dim=6144 --decoder-input-dim=1280 \
--decoder-embed-dim=1280 --decoder-output-dim=1280 \
--attention-dropout=0.1 --activation_dropout=0.1 --dropout=0.1 \
--max-tokens=2048 --tokens-per-sample=256 --sample-break-mode='eos' \
--warmup-updates=16000 --warmup-init-lr=1e-07 \
--lr_scheduler='inverse_sqrt' --lr=1 --max_epoch=250 --weight_decay=0.0 \
--seed=1 --log-format=json --log-interval=1000 --max-update=0 \
--save-interval-updates=10000 --keep-interval-update=1 --update_freq=1 \
--skip-invalid-size-inputs-valid-test
--ddp-backend="no_c10d" > lm_librispeech_word_transformer/train.log

# eval
cd [FAIRSEQ] && python eval_lm.py [MODEL_DST]/decoder/fairseq_word_data --gen-subset valid --sample-break-mode eos --seed 42 --tokens-per-sample 1024 --max-tokens=1024 --path lm_librispeech_word_transformer/checkpoint_best.pt
```
then we set `lr=0.05` and `max_epoch=100` and continue training with `inv_sqrt` policy.


## Results (Word-level perplexity, for wp model it is an upper limit)

| Language Model | params | dev-clean | dev-other |
|:-:|:-:|:-:|:-:|
| word 4-gram | - | 148 | 136.6 |
| word-pieces 6-gram | - | 145.4 | 133.7 |
| word-pieces GCNN | 188M | 61.7 | 61.9 |
| word-based GCNN | 319M | 57 | 57.9 |
| Transformer | 562M | 48.2 | 50.2 |
