#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

MODEL_DIR=$1
LANG=$2
BINUSER=mriviere
BINHOME=/private/home/$BINUSER

DECODE_BIN=$WAV2LETTERDIR/build/recipes/joint_training_vox_populi/cpc/Decode_cpc
ARCHDIR=$MODEL_DIR/arch

RUNNAME="${LANG}_nounsup"
RUNDIR="${MODEL_DIR}/fine_tuning"
NAME_MODEL="001_model_dev.bin"

UNSUP_LST="${COMMON_VOICE_DIR}/${LANG}/train_updated.lst"
TRAIN_LST="${COMMON_VOICE_DIR}/${LANG}/train_updated.lst"
VAL_LST="${COMMON_VOICE_DIR}/${LANG}/dev_updated.lst"
LEXICON="${COMMON_VOICE_DIR}/${LANG}/updated_lexicon.txt"
TOKENS="${COMMON_VOICE_DIR}/${LANG}/${LANG}_grapheme.tokens"


PATH_LOGS=${RUNDIR}/decoding_no_LM_${LANG}
mkdir $PATH_LOGS


$DECODE_BIN --am="${RUNDIR}/${NAME_MODEL}" \
            --test=${VAL_LST} \
            --lexicon=${LEXICON} \
            --tokens=${TOKENS}\
            --beamsize=250 \
            --uselexicon=true \
            --beamthreshold=40 \
            --logtostderr=1 \
            --features_type=raw \
            --sclite=$PATH_LOGS
