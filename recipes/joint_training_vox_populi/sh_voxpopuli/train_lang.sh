#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

MODEL_DIR=$1
LANG=$2
BINUSER=mriviere
BINHOME=/private/home/$BINUSER

PRETRAINEXE=$WAV2LETTERDIR/build/recipes/joint_training_vox_populi/cpc/Train_cpc
ARCHDIR=$MODEL_DIR/arch
PRETRAINRUN="003"

PRETRAINMODEL="${MODEL_DIR}/${PRETRAINRUN}_model_last.bin"

RUNNAME="${LANG}_nounsup"
RUNDIR="${MODEL_DIR}/fine_tuning"

UNSUP_LST="${COMMON_VOICE_DIR}/${LANG}/train_updated.lst"
TRAIN_LST="${COMMON_VOICE_DIR}/${LANG}/train_updated.lst"
VAL_LST="${COMMON_VOICE_DIR}/${LANG}/dev_updated.lst"
LEXICON="${COMMON_VOICE_DIR}/${LANG}/updated_lexicon.txt"
TOKENS="${COMMON_VOICE_DIR}/${LANG}/${LANG}_grapheme.tokens"

set -ex
export OMP_NUM_THREADS=1
export AF_MAX_BUFFERS=10000

echo "Running on $(hostname)"
JOBDIR="$(pwd)"

LR=0.00005
LRCRIT=0.000005
LR2=0.00005
LRCRIT2=0.000005
OPTIM="adam"
BETA1=0.9
BETA2=0.98
MOMENTUM=0.95
OPTIMEPSILON=1e-6
WEIGHTDECAY=0.01
MAXGRADNORM=25.
MAXGRADNORM2=0.1

WARMUP=12000
FREEZE=12000
SUPWARMUP=12000
HOLD=0
SUPHOLD=1000000
SUPDELAY=0
SUPDATES=1
LRSCHED="linear"
LRFINAL=0.00000005
LRSTEPDECAY=50000
UNSUPDATES=0
ITER=500000

BATCHING="dynamic"

ARCH="small"
FILTERBANKS=80
ENCODERDIM=512
CONTEXTDIM=768
MUTUALDIM=256
MASKLENGTH=10
MASKPROB=0.075
BATCHSIZE=6
MAXTOKENS=85000
MASKSAMETOKENPROB=0.0
MASKRANDTOKENPROB=0.0
MASKMIN=2
TEMPERATURE=0.1
NPIECES=320
NUNITS=2
NNEGATIVESAMPLES=100
NBUFFERSAMPLES=5
AUDIOSTRIDEMS=160
MININPUTMS=2000
MAXINPUTMS=33000
MAXCROP=15600
SAUG="false"
BIDIRECTIONAL="false"
PRETRAINV="dbg_pre16.4"
SUPER="libri"
PREENC="none"


TRAIN_CMD=$PRETRAINEXE

TRAIN_FLAGS=" \
               --rundir=$RUNDIR/$RUNNAME \
               --arch=$ARCHDIR/encoder_tr.arch,$ARCHDIR/context_tr.arch,$ARCHDIR/predict_tr.arch \
               --valid=dev:$VAL_LST \
               --train=$UNSUP_LST \
               --train2=$TRAIN_LST \
               --criterion=cpc \
               --criterion2=ctc \
               --pretrainmodel=$PRETRAINMODEL \
               --lr=$LR \
               --lrcrit=$LRCRIT \
               --lr2=$LR2 \
               --lrcrit2=$LRCRIT2 \
               --adambeta1=$BETA1 \
               --adambeta2=$BETA2 \
               --warmup=$WARMUP \
               --saug_warmup=$WARMUP \
               --saug_maskprob=0.025 \
               --supwarmup=$SUPWARMUP \
               --weightdecay=${WEIGHTDECAY} \
               --optimepsilon=${OPTIMEPSILON} \
               --eostoken=false \
               --netoptim=$OPTIM \
               --critoptim=$OPTIM \
               --momentum=${MOMENTUM} \
               --maxgradnorm=${MAXGRADNORM} \
               --maxgradnorm2=${MAXGRADNORM2} \
               --onorm=target \
               --replabel=0 \
               --wordseparator=| \
               --sqnorm \
               --nthread=0 \
               --batchsize=${BATCHSIZE} \
               --features_type=raw \
               --filterbanks=${FILTERBANKS} \
               --lexicon=$LEXICON\
               --tokens=$TOKENS \
               --codedim=${ENCODERDIM} \
               --contextdim=${CONTEXTDIM} \
               --mutualdim=${MUTUALDIM} \
               --masklength=${MASKLENGTH} \
               --maskprob=${MASKPROB} \
               --maskrandtokenprob=${MASKRANDTOKENPROB} \
               --masksametokenprob=${MASKSAMETOKENPROB} \
               --maskmin=${MASKMIN} \
               --temperature=${TEMPERATURE} \
               --npieces=${NPIECES} \
               --nunits=${NUNITS} \
               --nnegativesamples=${NNEGATIVESAMPLES} \
               --nbuffersamples=${NBUFFERSAMPLES} \
               --l2_enc_pen=0. \
               --iter=$ITER \
               --hold=$HOLD \
               --suphold=$SUPHOLD \
               --lr_sched=$LRSCHED \
               --lr_ld_final=$LRFINAL \
               --lr_step_decay=$LRSTEPDECAY \
               --freeze=$FREEZE \
               --unsupdates=$UNSUPDATES \
               --traincontext=true \
               --trainencoder=true \
               --batching_strategy=none \
               --batching_max_duration=$MAXTOKENS \
               --use_saug=true \
               --twostage=true \
               --supdates=$SUPDATES \
               --supdelay=0"
mkdir -p $RUNDIR


echo "DIRECTORY: $RUNDIR/$RUNNAME"

[ -d "$RUNDIR/$RUNNAME" ] || mkdir -p "$RUNDIR/$RUNNAME"

if [ -f "$RUNDIR/$RUNNAME/001_model_last.bin" ]; then
    $TRAIN_CMD continue $RUNDIR/$RUNNAME $TRAIN_FLAGS $DISTRIBUTED_FLAGS
else
    $TRAIN_CMD train $TRAIN_FLAGS $DISTRIBUTED_FLAGS
fi
