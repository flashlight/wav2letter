#!/bin/bash
MODEL_DST="$1"

wrdlm=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_word_4g
char5gram=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_char_5g
char10gram=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_char_10g
char15gramp=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_char_15g_pruned
char15gram=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_char_15g
char20gramp=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_char_20g_pruned
char20gram=$MODEL_DST/decoder/ngram_models/lm_wsj_kenlm_char_20g
vocabfile=$MODEL_DST/decoder/kenlm_limit_vocab_file.txt


# word 4gram model
echo "Words perplexity for word 4gram model"
echo "Nov93dev"
tail -n3 "$wrdlm.nov93dev" | head -n2

echo "Words perplexity upper limit for char 5gram model"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char5gram.nov93dev"

echo "Words perplexity upper limit for char 10gram model"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char10gram.nov93dev"

echo "Words perplexity upper limit for char pruned 15gram model"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char15gramp.nov93dev"

echo "Words perplexity upper limit for char 15gram model"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char15gram.nov93dev"

echo "Words perplexity upper limit for char pruned 20gram model"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char20gramp.nov93dev"

echo "Words perplexity upper limit for char 20gram model"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char20gram.nov93dev"

echo "Words perplexity lower limit (also with upper limit to recheck) for char 5gram model"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.nov93dev" --word_model "$wrdlm.bin" --char_model "$char5gram.bin" > "$char5gram.nov93dev.ppl.log"

echo "Words perplexity lower limit (also with upper limit to recheck) for char 10gram model"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.nov93dev" --word_model "$wrdlm.bin" --char_model "$char10gram.bin" > "$char10gram.nov93dev.ppl.log"

echo "Words perplexity lower limit (also with upper limit to recheck) for char pruned 15gram model"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.nov93dev" --word_model "$wrdlm.bin" --char_model "$char15gramp.bin" > "$char15gramp.nov93dev.ppl.log"

echo "Words perplexity lower limit (also with upper limit to recheck) for char 15gram model"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.nov93dev" --word_model "$wrdlm.bin" --char_model "$char15gram.bin" > "$char15gram.nov93dev.ppl.log"

echo "Words perplexity lower limit (also with upper limit to recheck) for char pruned 20gram model"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.nov93dev" --word_model "$wrdlm.bin" --char_model "$char20gramp.bin" > "$char20gramp.nov93dev.ppl.log"

echo "Words perplexity lower limit (also with upper limit to recheck) for char 20gram model"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.nov93dev" --word_model "$wrdlm.bin" --char_model "$char20gram.bin" > "$char20gram.nov93dev.ppl.log"
