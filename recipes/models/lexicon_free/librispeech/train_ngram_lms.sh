#!/bin/bash
DATA_DST="$1"
MODEL_DST="$2"
KENLM="$3"

mkdir -p "$MODEL_DST/decoder/ngram_models/"
wrdlm=$MODEL_DST/decoder/ngram_models/lm_librispeech_kenlm_word_4g_200kvocab
char5gram=$MODEL_DST/decoder/ngram_models/lm_librispeech_kenlm_char_5g
char10gram=$MODEL_DST/decoder/ngram_models/lm_librispeech_kenlm_char_10g
char15gram=$MODEL_DST/decoder/ngram_models/lm_librispeech_kenlm_char_15g_pruned
char17gram=$MODEL_DST/decoder/ngram_models/lm_librispeech_kenlm_char_17g_pruned
char20gram=$MODEL_DST/decoder/ngram_models/lm_librispeech_kenlm_char_20g_pruned

# word 4gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback -o 4 \
 --limit_vocab_file "$MODEL_DST/decoder/kenlm_limit_vocab_file.txt" trie < "$DATA_DST/text/librispeech-lm-norm.txt.lower.shuffle" > "$wrdlm.arpa"
"$KENLM/build_binary" trie "$wrdlm.arpa" "$wrdlm.bin"
"$KENLM/query" "$wrdlm.bin" <"$DATA_DST/text/librispeech-lm-norm.txt.lower.shuffle" > "$wrdlm.train"
"$KENLM/query" "$wrdlm.bin" <"$DATA_DST/text/dev-clean.txt" > "$wrdlm.dev-clean"
"$KENLM/query" "$wrdlm.bin" <"$DATA_DST/text/dev-other.txt" > "$wrdlm.dev-other"

# char 5gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback -o 5 trie < "$MODEL_DST/decoder/char_lm_data.train" > "$char5gram.arpa"
"$KENLM/build_binary" trie "$char5gram.arpa" "$char5gram.bin"
"$KENLM/query" "$char5gram.bin" <"$MODEL_DST/decoder/char_lm_data.train" > "$char5gram.train"
"$KENLM/query" "$char5gram.bin" <"$MODEL_DST/decoder/char_lm_data.dev-clean" > "$char5gram.dev-clean"
"$KENLM/query" "$char5gram.bin" <"$MODEL_DST/decoder/char_lm_data.dev-other" > "$char5gram.dev-other"

# char 10gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback -o 10 trie < "$MODEL_DST/decoder/char_lm_data.train" > "$char10gram.arpa"
"$KENLM/build_binary" trie "$char10gram.arpa" "$char10gram.bin"
"$KENLM/query" "$char10gram.bin" <"$MODEL_DST/decoder/char_lm_data.train" > "$char10gram.train"
"$KENLM/query" "$char10gram.bin" <"$MODEL_DST/decoder/char_lm_data.dev-clean" > "$char10gram.dev-clean"
"$KENLM/query" "$char10gram.bin" <"$MODEL_DST/decoder/char_lm_data.dev-other" > "$char10gram.dev-other"

# NOTE: Further models needs a lot of space on the disk
# char 15gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback --prune 0 0 0 0 0 1 1 1 2 3 -o 15 trie < "$MODEL_DST/decoder/char_lm_data.train" > "$char15gram.arpa"
"$KENLM/build_binary" trie "$char15gram.arpa" "$char15gram.bin"
"$KENLM/query" "$char15gram.bin" <"$MODEL_DST/decoder/char_lm_data.train" > "$char15gram.train"
"$KENLM/query" "$char15gram.bin" <"$MODEL_DST/decoder/char_lm_data.dev-clean" > "$char15gram.dev-clean"
"$KENLM/query" "$char15gram.bin" <"$MODEL_DST/decoder/char_lm_data.dev-other" > "$char15gram.dev-other"


# char 17gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback --prune 0 0 0 0 0 1 1 1 2 3 -o 17 trie < "$MODEL_DST/decoder/char_lm_data.train" > "$char17gram.arpa"
"$KENLM/build_binary" trie "$char17gram.arpa" "$char17gram.bin"
"$KENLM/query" "$char17gram.bin" <"$MODEL_DST/decoder/char_lm_data.train" > "$char17gram.train"
"$KENLM/query" "$char17gram.bin" <"$MODEL_DST/decoder/char_lm_data.dev-clean" > "$char17gram.dev-clean"
"$KENLM/query" "$char17gram.bin" <"$MODEL_DST/decoder/char_lm_data.dev-other" > "$char17gram.dev-other"


# char 20gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback --prune 0 0 0 0 0 1 1 1 2 3 -o 20 trie < "$MODEL_DST/decoder/char_lm_data.train" > "$char20gram.arpa"
"$KENLM/build_binary" trie "$char20gram.arpa" "$char20gram.bin"
"$KENLM/query" "$char20gram.bin" <"$MODEL_DST/decoder/char_lm_data.train" > "$char20gram.train"
"$KENLM/query" "$char20gram.bin" <"$MODEL_DST/decoder/char_lm_data.dev-clean" > "$char20gram.dev-clean"
"$KENLM/query" "$char20gram.bin" <"$MODEL_DST/decoder/char_lm_data.dev-other" > "$char20gram.dev-other"
