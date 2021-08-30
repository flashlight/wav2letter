#!/bin/bash
DATA_DST="$1"
MODEL_DST="$2"
KENLM="$3"

mkdir -p "$MODEL_DST/decoder/ngram_models/"
wrdlm=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_word_4g
char5gram=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_char_5g
char10gram=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_char_10g
char15gramp=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_char_15g_pruned
char15gram=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_char_15g
char20gramp=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_char_20g_pruned
char20gram=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_char_20g

# word 4gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback -o 4 trie < "$DATA_DST/text/lm.txt" > "$wrdlm.arpa"
"$KENLM/build_binary" trie "$wrdlm.arpa" "$wrdlm.bin"
"$KENLM/query" "$wrdlm.bin" <"$DATA_DST/text/lm.txt" > "$wrdlm.train"
"$KENLM/query" "$wrdlm.bin" <"$MODEL_DST/decoder/word_lm_data.nov93dev" > "$wrdlm.nov93dev"

# char 5gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback -o 5 trie < "$MODEL_DST/decoder/char_lm_data.train" > "$char5gram.arpa"
"$KENLM/build_binary" trie "$char5gram.arpa" "$char5gram.bin"
"$KENLM/query" "$char5gram.bin" <"$MODEL_DST/decoder/char_lm_data.train" > "$char5gram.train"
"$KENLM/query" "$char5gram.bin" <"$MODEL_DST/decoder/char_lm_data.nov93dev" > "$char5gram.nov93dev"

# char 10gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback -o 10 trie < "$MODEL_DST/decoder/char_lm_data.train" > "$char10gram.arpa"
"$KENLM/build_binary" trie "$char10gram.arpa" "$char10gram.bin"
"$KENLM/query" "$char10gram.bin" <"$MODEL_DST/decoder/char_lm_data.train" > "$char10gram.train"
"$KENLM/query" "$char10gram.bin" <"$MODEL_DST/decoder/char_lm_data.nov93dev" > "$char10gram.nov93dev"

# char 15gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback --prune 0 0 0 0 0 1 1 1 2 3 -o 15 trie < "$MODEL_DST/decoder/char_lm_data.train" > "$char15gramp.arpa"
"$KENLM/build_binary" trie "$char15gramp.arpa" "$char15gramp.bin"
"$KENLM/query" "$char15gramp.bin" <"$MODEL_DST/decoder/char_lm_data.train" > "$char15gramp.train"
"$KENLM/query" "$char15gramp.bin" <"$MODEL_DST/decoder/char_lm_data.nov93dev" > "$char15gramp.nov93dev"

# char 15gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback -o 15 trie < "$MODEL_DST/decoder/char_lm_data.train" > "$char15gram.arpa"
"$KENLM/build_binary" trie "$char15gram.arpa" "$char15gram.bin"
"$KENLM/query" "$char15gram.bin" <"$MODEL_DST/decoder/char_lm_data.train" > "$char15gram.train"
"$KENLM/query" "$char15gram.bin" <"$MODEL_DST/decoder/char_lm_data.nov93dev" > "$char15gram.nov93dev"

# char 20gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback --prune 0 0 0 0 0 1 1 1 2 3 -o 20 trie < "$MODEL_DST/decoder/char_lm_data.train" > "$char20gramp.arpa"
"$KENLM/build_binary" trie "$char20gramp.arpa" "$char20gramp.bin"
"$KENLM/query" "$char20gramp.bin" <"$MODEL_DST/decoder/char_lm_data.train" > "$char20gramp.train"
"$KENLM/query" "$char20gramp.bin" <"$MODEL_DST/decoder/char_lm_data.nov93dev" > "$char20gramp.nov93dev"

# char 20gram model
"$KENLM/lmplz" -T /tmp -S 50G --discount_fallback -o 20 trie < "$MODEL_DST/decoder/char_lm_data.train" > "$char20gram.arpa"
"$KENLM/build_binary" trie "$char20gram.arpa" "$char20gram.bin"
"$KENLM/query" "$char20gram.bin" <"$MODEL_DST/decoder/char_lm_data.train" > "$char20gram.train"
"$KENLM/query" "$char20gram.bin" <"$MODEL_DST/decoder/char_lm_data.nov93dev" > "$char20gram.nov93dev"
