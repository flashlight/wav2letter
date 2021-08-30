#!/bin/bash
DATA_DST="$1"
MODEL_DST="$2"

wrdlm=$MODEL_DST/decoder/convlm_models/word_14B/checkpoint_best.pt
charlm=$MODEL_DST/decoder/convlm_models/char_14B/checkpoint_best.pt
charlm20B=$MODEL_DST/decoder/convlm_models/char_20B/checkpoint_best.pt

# word 4gram model
echo "Words perplexity for word convlm"
python3 ../utilities/compute_upper_ppl_convlm.py --model "$wrdlm" --dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" --text "$DATA_DST/text/nov93dev.txt" --model_type word --dataset_type="wsj"

echo "Upper limit on words perplexity for char 14B convlm"
python3 ../utilities/compute_upper_ppl_convlm.py --model "$charlm" --dict "$MODEL_DST/decoder/fairseq_char_data/dict.txt" \
  --word_dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" --text "$MODEL_DST/decoder/char_lm_data.nov93dev" --model_type char14B --dataset_type="wsj"

echo "Upper limit on words perplexity for char 20B convlm"
python3 ../utilities/compute_upper_ppl_convlm.py --model "$charlm20B" --dict "$MODEL_DST/decoder/fairseq_char_data/dict.txt" \
  --word_dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" --text "$MODEL_DST/decoder/char_lm_data.nov93dev" --model_type char20B --dataset_type="wsj"

echo "Upper limit on words perplexity for char 14B convlm"
python3 ../utilities/compute_lower_ppl_convlm.py \
  --model "$charlm" --dict "$MODEL_DST/decoder/fairseq_char_data/dict.txt" \
  --word_model "$wrdlm" --word_dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" \
  --text "$MODEL_DST/decoder/char_lm_data.nov93dev" --model_type char14B > "$charlm.nov93dev.ppl.log" --dataset_type="wsj"

echo "Upper limit on words perplexity for char 20B convlm"
python3 ../utilities/compute_lower_ppl_convlm.py \
  --model "$charlm20B" --dict "$MODEL_DST/decoder/fairseq_char_data/dict.txt" \
  --word_model "$wrdlm" --word_dict "$MODEL_DST/decoder/fairseq_word_data/dict.txt" \
  --text "$MODEL_DST/decoder/char_lm_data.nov93dev" --model_type char20B > "$charlm20B.nov93dev.ppl.log" --dataset_type="wsj"
