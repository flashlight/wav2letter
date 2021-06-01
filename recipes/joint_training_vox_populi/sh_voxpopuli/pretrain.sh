#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
GPUS=$1
RUNDIR=$2
VOX_POPULI_100k_lst=$3
VAL_LST=$4
SUPDIR=$5
GDB=${GDB:=0}

echo "Running on $(hostname)"
JOBDIR="$(pwd)"

PRETRAINEXE=$WAV2LETTERDIR/build/recipes/joint_training_vox_populi/cpc/Train_cpc
ARCHDIR=$WAV2LETTERDIR/recipes/joint_training_vox_populi/small
RUNNAME="pretrain_vp_100k"


# If twostage=false then a part of the training is supervised
# and you must add the corresponding lst for the supervised data
# as well as the lexicon and the token files
# If twostage=true, put any valid lst, tokens and lexicon here,
# it doesn't matter.
SUPLIST="${SUPDIR}/data.lst"
LEXICON="${SUPDIR}/lexicon.txt"
TOKENS="${SUPDIR}/grapheme.tokens"


# unsupervised training
LR=0.0005
LRCRIT=0.0005
OPTIM="adam"
BETA1=0.9
BETA2=0.98
MOMENTUM=0.95
OPTIMEPSILON=1e-6
WEIGHTDECAY=0.01
MAXGRADNORM=25.
WARMUP=240000
HOLD=0
UNSUPITER=20000000

# supervised training
LR2=0.000025
LRCRIT2=0.000025
MAXGRADNORM2=1.
SUPWARMUP=240000
SUPHOLD=1000000
FREEZE=0
TOTALITER=20000000


# ~100M param model
ARCH="small"
ENCODERDIM=512
CONTEXTDIM=768
MUTUALDIM=256
MASKLENGTH=10
MASKPROB=0.075
BATCHSIZE=3
MAXTOKENS=87500

# ~300M param model
#ARCH="big"
#ENCODERDIM=512
#CONTEXTDIM=1024
#MUTUALDIM=768
#MASKLENGTH=10
#MASKPROB=0.065
#BATCHSIZE=2
#MAXTOKENS=40000

# common params
UNSUPDATES=1
SUPDATES=1
LRSCHED="linear"
LRFINAL=0.1
LRSTEPDECAY=50000
BATCHING="dynamic"
#BATCHING="none"

FILTERBANKS=80
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
PRETRAINV="dbg2"
SUPER="libri"
PREENC="none"

TRAIN_CMD=$PRETRAINEXE

TRAIN_FLAGS=" \
               --rundir=$RUNDIR/$RUNNAME \
               --arch=$ARCHDIR/encoder_tr.arch,$ARCHDIR/context_tr.arch,$ARCHDIR/predict_tr.arch \
               --valid=dev-clean:$DATADIR/dev-clean.lst,dev-other:$DATADIR/dev-other.lst \
               --train=$UNSUPLIST \
               --train2=$SUPLIST \
               --features_type=raw \
               --rndv_filepath=${RNDVDIR} \
               --criterion=cpc \
               --criterion2=ctc \
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
               --netoptim=$OPTIM \
               --critoptim=$OPTIM \
               --momentum=${MOMENTUM} \
               --maxgradnorm=${MAXGRADNORM} \
               --maxgradnorm2=${MAXGRADNORM2} \
               --onorm=target \
               --replabel=0 \
               --surround=| \
               --sqnorm \
               --nthread=0 \
               --batchsize=${BATCHSIZE} \
               --filterbanks=${FILTERBANKS} \
               --lexicon=$DATADIR3/lexicon_train+dev.txt \
               --tokens=$DATADIR3/tokens.txt \
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
               --iter=$TOTALITER \
               --hold=$HOLD \
               --suphold=$SUPHOLD \
               --lr_sched=$LRSCHED \
               --lr_ld_final=$LRFINAL \
               --lr_step_decay=$LRSTEPDECAY \
               --freeze=$FREEZE \
               --unsupdates=$UNSUPDATES \
               --traincontext=true \
               --trainencoder=true \
               --batching_strategy=$BATCHING \
               --batching_max_duration=$MAXTOKENS \
               --use_saug=true \
               --twostage=true \
               --reportiters=10000 \
               --itersave=true \
               --supdates=$SUPDATES \
               --supdelay=$UNSUPITER"

mkdir -p $RUNDIR

echo "DIRECTORY: $RUNDIR/$RUNNAME"

[ -d "$RUNDIR/$RUNNAME" ] || mkdir -p "$RUNDIR/$RUNNAME"

if [ -f "$RUNDIR/$RUNNAME/001_model_last.bin" ]; then
    if [ $GDB -eq 0 ]; then
        $TRAIN_CMD continue $RUNDIR/$RUNNAME $TRAIN_FLAGS $MPI_FLAGS
    else
        gdb --args $TRAIN_CMD continue $RUNDIR/$RUNNAME $TRAIN_FLAGS $MPI_FLAGS
    fi
else
    if [ $GDB -eq 0 ]; then
        $TRAIN_CMD train $TRAIN_FLAGS $MPI_FLAGS
    else
        gdb --args $TRAIN_CMD train $TRAIN_FLAGS $MPI_FLAGS
    fi
fi
