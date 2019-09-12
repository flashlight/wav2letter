#!/bin/bash
DATA_DST="$1"
MODEL_DST="$2"
FAIRSEQ="$3"

mkdir -p "$MODEL_DST/decoder/fairseq_word_data/"
mkdir -p "$MODEL_DST/decoder/fairseq_char_data/"

python "$FAIRSEQ/preprocess.py" --only-source \
--trainpref "$DATA_DST/text/librispeech-lm-norm.txt.lower.shuffle" \
--validpref "$DATA_DST/text/dev-clean.txt" \
--testpref "$DATA_DST/text/dev-other.txt" \
--destdir "$MODEL_DST/decoder/fairseq_word_data/" \
--thresholdsrc 10 \
--padding-factor 1 \
--workers 16

cut -f1 -d " " "$MODEL_DST/decoder/fairseq_word_data/dict.txt" | tr "\n" " " > "$MODEL_DST/decoder/kenlm_limit_vocab_file.txt"

python "$FAIRSEQ/preprocess.py" --only-source \
--trainpref "$MODEL_DST/decoder/char_lm_data.train" \
--validpref "$MODEL_DST/decoder/char_lm_data.dev-clean" \
--testpref "$MODEL_DST/decoder/char_lm_data.dev-other" \
--destdir "$MODEL_DST/decoder/fairseq_char_data/" \
--thresholdsrc 10 \
--workers 16
