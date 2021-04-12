#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

LANG=$1
DIR_CV_LANG="${COMMON_VOICE_DIR}/${LANG}"
PATH_TOKENS="${DIR_CV_LANG}/${LANG}_grapheme.tokens"
PATH_CLIPS_MP3="${DIR_CV_LANG}/clips"
PATH_CLIPS_FLAC="${DIR_CV_LANG}/validated_flac"
PATH_LEXICON="${DIR_CV_LANG}/updated_lexicon.txt"


# Start by building the tokens
echo "Building $PATH_TOKENS"
python get_tokens.py ${DIR_CV_LANG}/validated.tsv ${PATH_TOKENS}

# Make the lst conversion
python common_voice_to_wav2letter.py --path_tsv ${DIR_CV_LANG}/validated.tsv \
                                     --path_audio ${PATH_CLIPS_MP3} \
                                     --path_tokens ${PATH_TOKENS} \
                                     --path_conversion ${PATH_CLIPS_FLAC} \
                                     --file_extension .mp3 \
                                     --path_output ${DIR_CV_LANG}/validated_updated.lst


# # Build dev.lst
echo "Building ${DIR_CV_LANG}/dev.lst"
python common_voice_to_wav2letter.py --path_tsv ${DIR_CV_LANG}/dev.tsv \
                                     --path_audio ${PATH_CLIPS_MP3} \
                                     --path_tokens ${PATH_TOKENS} \
                                     --path_conversion ${PATH_CLIPS_FLAC} \
                                     --file_extension .mp3 \
                                     --path_output ${DIR_CV_LANG}/dev_updated.lst


# Build test.lst
echo "Building ${DIR_CV_LANG}/test.lst"
python common_voice_to_wav2letter.py --path_tsv ${DIR_CV_LANG}/test.tsv \
                                     --path_audio ${PATH_CLIPS_MP3} \
                                     --path_tokens ${PATH_TOKENS} \
                                     --path_conversion ${PATH_CLIPS_FLAC} \
                                     --file_extension .mp3 \
                                     --path_output ${DIR_CV_LANG}/test_updated.lst

# Build train.lst
echo "Building ${DIR_CV_LANG}/train.lst"
python common_voice_to_wav2letter.py --path_tsv ${DIR_CV_LANG}/train.tsv \
                                     --path_audio ${PATH_CLIPS_MP3} \
                                     --path_tokens ${PATH_TOKENS} \
                                     --path_conversion ${PATH_CLIPS_FLAC} \
                                     --file_extension .mp3 \
                                     --path_output ${DIR_CV_LANG}/train_updated.lst


python make_lexicon.py -i ${DIR_CV_LANG}/validated_updated.lst \
                       --tokens $PATH_TOKENS \
                       -o $PATH_LEXICON \
                       --min_occ 0 \
                       --max_size_lexicon 100000000000000
