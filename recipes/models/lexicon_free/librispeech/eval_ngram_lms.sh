#!/bin/bash
MODEL_DST="$1"

wrdlm=$MODEL_DST/decoder/ngram_models/lm_librispeech_kenlm_word_4g_200kvocab
char5gram=$MODEL_DST/decoder/ngram_models/lm_librispeech_kenlm_char_5g
char10gram=$MODEL_DST/decoder/ngram_models/lm_librispeech_kenlm_char_10g
char15gram=$MODEL_DST/decoder/ngram_models/lm_librispeech_kenlm_char_15g_pruned
char17gram=$MODEL_DST/decoder/ngram_models/lm_librispeech_kenlm_char_17g_pruned
char20gram=$MODEL_DST/decoder/ngram_models/lm_librispeech_kenlm_char_20g_pruned
vocabfile=$MODEL_DST/decoder/kenlm_limit_vocab_file.txt

# word 4gram model
echo "Words perplexity for word 4gram model"
echo "Dev clean"
tail -n3 "$wrdlm.dev-clean" | head -n2
echo "Dev other"
tail -n3 "$wrdlm.dev-other" | head -n2

echo "Words perplexity upper limit for char 5gram model"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char5gram.dev-clean"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char5gram.dev-other"

echo "Words perplexity upper limit for char 10gram model"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char10gram.dev-clean"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char10gram.dev-other"

echo "Words perplexity upper limit for char 15gram model"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char15gram.dev-clean"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char15gram.dev-other"

echo "Words perplexity upper limit for char 17gram model"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char17gram.dev-clean"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char17gram.dev-other"

echo "Words perplexity upper limit for char 20gram model"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char20gram.dev-clean"
python3 ../utilities/compute_upper_ppl_kenlm.py --vocab_file "$vocabfile" --kenlm_preds "$char20gram.dev-other"

echo "Words perplexity lower limit (also with upper limit to recheck) for char 5gram model"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.dev-clean" --word_model "$wrdlm.bin" --char_model "$char5gram.bin" > "$char5gram.dev-clean.ppl.log"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.dev-other" --word_model "$wrdlm.bin" --char_model "$char5gram.bin" > "$char5gram.dev-other.ppl.log"

echo "Words perplexity lower limit (also with upper limit to recheck) for char 10gram model"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.dev-clean" --word_model "$wrdlm.bin" --char_model "$char10gram.bin" > "$char10gram.dev-clean.ppl.log"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.dev-other" --word_model "$wrdlm.bin" --char_model "$char10gram.bin" > "$char10gram.dev-other.ppl.log"

echo "Words perplexity lower limit (also with upper limit to recheck) for char 15gram model"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.dev-clean" --word_model "$wrdlm.bin" --char_model "$char15gram.bin" > "$char15gram.dev-clean.ppl.log"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.dev-other" --word_model "$wrdlm.bin" --char_model "$char15gram.bin" > "$char15gram.dev-other.ppl.log"

echo "Words perplexity lower limit (also with upper limit to recheck) for char 17gram model"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.dev-clean" --word_model "$wrdlm.bin" --char_model "$char17gram.bin" > "$char17gram.dev-clean.ppl.log"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.dev-other" --word_model "$wrdlm.bin" --char_model "$char17gram.bin" > "$char17gram.dev-other.ppl.log"

echo "Words perplexity lower limit (also with upper limit to recheck) for char 20gram model"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.dev-clean" --word_model "$wrdlm.bin" --char_model "$char20gram.bin" > "$char20gram.dev-clean.ppl.log"
python3 ../utilities/compute_lower_ppl_kenlm.py --vocab_file "$vocabfile" --text "$MODEL_DST/decoder/char_lm_data.dev-other" --word_model "$wrdlm.bin" --char_model "$char20gram.bin" > "$char20gram.dev-other.ppl.log"
