#!/bin/bash

set -ex
export OMP_NUM_THREADS=1
export AF_MAX_BUFFERS=10000

GDB=${GDB:=0}

echo "Running on $(hostname)"
JOBDIR="$(pwd)"
GPUS=${GPUS:=64}

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
WARMUP=20000
HOLD=0
UNSUPITER=250000

# supervised training
LR2=0.000025
LRCRIT2=0.000025
MAXGRADNORM2=1.
SUPWARMUP=20000
SUPHOLD=1000000
FREEZE=0
TOTALITER=400000


# ~100M param model
ARCH="small"
ENCODERDIM=512
CONTEXTDIM=768
MUTUALDIM=256
MASKLENGTH=10
MASKPROB=0.075
BATCHSIZE=6
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
PRETRAINEXE=Train_${PRETRAINV}
PREENC="none"

CHECKINPUTMS=$(($AUDIOSTRIDEMS * ($NNEGATIVESAMPLES + 2*$NBUFFERSAMPLES + 1 )))
#if (( $MININPUTMS < $CHECKINPUTMS )); then
#  echo "MININPUTMS ($MININPUTMS) too small, need at least $CHECKINPUTMS"
#  exit 1
#fi

SRCDIR="$HOME/sources/wav2letter"
DATADIR="/path/to/datadir/"
RUNDIR="/path/to/rundir/"
ARCHDIR=$SRCDIR/recipes/joint_training/$ARCH

RUNNAME="${PRETRAINV}_${PREENC}_tr_${ARCH}_${SUPER}_optim${OPTIM}_lr${LR}_lrcrit${LRCRIT}_momentum${MOMENTUM}_mgn${MAXGRADNORM}_warmup${WARMUP}_fb${FILTERBANKS}_encoder${ENCODERDIM}_context${CONTEXTDIM}_mutual${MUTUALDIM}_masklength${MASKLENGTH}_maskprob${MASKPROB}_temp${TEMPERATURE}_nneg${NNEGATIVESAMPLES}_nbuff${NBUFFERSAMPLES}_saug${SAUG}_bsz${BATCHSIZE}_unsup${UNSUPDATES}.${SUPDATES}_iter${UNSUPITER}.${TOTALITER}_ngpus64"

TRAIN_CMD=$SRCDIR/build/Train_cpc
UNSUPLIST=$DATADIR/train-clean-100.lst,$DATADIR/train-clean-360.lst,$DATADIR/train-other-500.lst
SUPLIST=$DATADIR/train-clean-100.lst

TRAIN_FLAGS=" \
               --rundir=$RUNDIR/$RUNNAME \
               --arch=$ARCHDIR/encoder_tr.arch,$ARCHDIR/context_tr.arch,$ARCHDIR/predict_tr.arch \
               --valid=dev-clean:$DATADIR/dev-clean.lst,dev-other:$DATADIR/dev-other.lst \
               --train=$UNSUPLIST \
               --train2=$SUPLIST \
               --criterion=cpc \
               --criterion2=ctc \
               --lr=$LR \
               --lrcrit=$LRCRIT \
               --lr2=$LR2 \
               --lrcrit2=$LRCRIT2 \
               --adambeta1="$BETA1" \
               --adambeta2="$BETA2" \
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
               --surround=| \
               --sqnorm \
               --nthread=0 \
               --batchsize=${BATCHSIZE} \
               --filterbanks=${FILTERBANKS} \
               --lexicon=$DATADIR/lexicon_train+dev.txt \
               --tokens=$DATADIR/tokens.txt \
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
               --supdates=$SUPDATES \
               --supdelay=$UNSUPITER"

mkdir -p $RUNDIR

if (( GPUS > 1 )); then
    export NCCL_LL_THRESHOLD=0
    TRAIN_CMD="mpirun -x AF_MAX_BUFFERS=$AF_MAX_BUFFERS -n ${GPUS} $TRAIN_CMD"
    MPI_FLAGS="--enable_distributed --world_size=${GPUS}"
fi

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

