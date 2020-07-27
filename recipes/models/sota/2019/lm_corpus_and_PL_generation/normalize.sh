#!/bin/bash
cat $1 | python3 dump.py | python3 preprocessing.py \
        | perl "mosesdecoder/scripts/tokenizer/normalize-punctuation.perl" \
        | perl "mosesdecoder/scripts/ems/support/split-sentences.perl" \
        | python3 skip_paragraph.py \
        | perl "mosesdecoder/scripts/tokenizer/tokenizer.perl" -no-escape \
        | python3 postprocessing.py > "librispeech_lm_corpus_raw_without_librivox.txt.norm"
