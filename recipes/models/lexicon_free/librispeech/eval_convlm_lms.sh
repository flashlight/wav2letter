#!/bin/bash
DATA_DST="$1"
MODEL_DST="$2"

wrdlm=$MODEL_DST/decoder/convlm_models/word_14B/checkpoint_best.pt
charlm=$MODEL_DST/decoder/convlm_models/char_14B/checkpoint_best.pt
charlm20B=$MODEL_DST/decoder/convlm_models/char_20B/checkpoint_best.pt

# word 4gram model
echo "Words perplexity for word convlm"
python3 ../utilities/compute_upper_ppl_convlm.py --model "$wrdlm" --dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" --text "$DATA_DST/text/dev-clean.txt" --model_type word --dataset_type="ls"
python3 ../utilities/compute_upper_ppl_convlm.py --model "$wrdlm" --dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" --text "$DATA_DST/text/dev-other.txt" --model_type word --dataset_type="ls"

echo "Upper limit on words perplexity for char 14B convlm"
python3 ../utilities/compute_upper_ppl_convlm.py --model "$charlm" --dict "$MODEL_DST/decoder/fairseq_char_data/dict.txt" \
  --word_dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" --text "$MODEL_DST/decoder/char_lm_data.dev-clean" --model_type char14B --dataset_type="ls"
python3 ../utilities/compute_upper_ppl_convlm.py --model "$charlm" --dict "$MODEL_DST/decoder/fairseq_char_data/dict.txt" \
  --word_dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" --text "$MODEL_DST/decoder/char_lm_data.dev-other" --model_type char14B --dataset_type="ls"

echo "Upper limit on words perplexity for char 20B convlm"
python3 ../utilities/compute_upper_ppl_convlm.py --model "$charlm20B" --dict "$MODEL_DST/decoder/fairseq_char_data/dict.txt" \
  --word_dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" --text "$MODEL_DST/decoder/char_lm_data.dev-clean" --model_type char20B --dataset_type="ls"
python3 ../utilities/compute_upper_ppl_convlm.py --model "$charlm20B" --dict "$MODEL_DST/decoder/fairseq_char_data/dict.txt" \
  --word_dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" --text "$MODEL_DST/decoder/char_lm_data.dev-other" --model_type char20B --dataset_type="ls"

echo "Upper amd lower limits on words perplexity for char 14B convlm"
python3 ../utilities/compute_lower_ppl_convlm.py \
  --model "$charlm" --dict "$MODEL_DST/decoder/fairseq_char_data/dict.txt" \
  --word_model "$wrdlm" --word_dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" \
  --text "$MODEL_DST/decoder/char_lm_data.dev-clean" --model_type char14B > "$charlm.dev-clean.ppl.log" --dataset_type="ls"
python3 ../utilities/compute_lower_ppl_convlm.py \
  --model "$charlm" --dict "$MODEL_DST/decoder/fairseq_char_data/dict.txt" \
  --word_model "$wrdlm" --word_dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" \
  --text "$MODEL_DST/decoder/char_lm_data.dev-other" --model_type char14B > "$charlm.dev-other.ppl.log" --dataset_type="ls"

echo "Upper and lower limits on words perplexity for char 20B convlm"
python3 ../utilities/compute_lower_ppl_convlm.py \
  --model "$charlm20B" --dict "$MODEL_DST/decoder/fairseq_char_data/dict.txt" \
  --word_model "$wrdlm" --word_dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" \
  --text "$MODEL_DST/decoder/char_lm_data.dev-clean" --model_type char20B > "$charlm20B.dev-clean.ppl.log" --dataset_type="ls"
python3 ../utilities/compute_lower_ppl_convlm.py \
  --model "$charlm20B" --dict "$MODEL_DST/decoder/fairseq_char_data/dict.txt" \
  --word_model "$wrdlm" --word_dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" \
  --text "$MODEL_DST/decoder/char_lm_data.dev-other" --model_type char20B > "$charlm20B.dev-other.ppl.log" --dataset_type="ls"
