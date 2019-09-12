#!/bin/bash
MODEL_DST="$1"
WAV2LETTER="$2"

printf "<fairseq_style>\n<pad>\n</s>\n<unk>\n" > "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_word_14B.vocab"
cut -f1 -d " " "$MODEL_DST/decoder/fairseq_word_data/dict.txt" >> "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_word_14B.vocab"
printf "<fairseq_style>\n<pad>\n</s>\n<unk>\n" > "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_char_20B.vocab"
cut -f1 -d " " "$MODEL_DST/decoder/fairseq_char_data/dict.txt" >> "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_char_20B.vocab"
python3 ../../utilities/convlm_serializer/save_pytorch_model.py "$MODEL_DST/decoder/convlm_models/word_14B/checkpoint_best.pt" "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_word_14B.weights"
python3 ../../utilities/convlm_serializer/save_pytorch_model.py "$MODEL_DST/decoder/convlm_models/char_14B/checkpoint_best.pt" "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_char_14B.weights"
python3 ../../utilities/convlm_serializer/save_pytorch_model.py "$MODEL_DST/decoder/convlm_models/char_20B/checkpoint_best.pt" "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_char_20B.weights"
"$WAV2LETTER/build/recipes/models/utilities/convlm_serializer/SerializeConvLM" \
  lm_wsj_convlm_word_14B.arch \
  "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_word_14B.weights" \
  "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_word_14B.bin" \
  162464 0 1 10000,50000,100000 4096
"$WAV2LETTER/build/recipes/models/utilities/convlm_serializer/SerializeConvLM" \
  lm_wsj_convlm_char_14B.arch \
  "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_char_14B.weights" \
  "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_char_14B.bin" 40 1
"$WAV2LETTER/build/recipes/models/utilities/convlm_serializer/SerializeConvLM" \
  lm_wsj_convlm_char_20B.arch \
  "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_char_20B.weights" \
  "$MODEL_DST/decoder/convlm_models/lm_wsj_convlm_char_20B.bin" 40 1
