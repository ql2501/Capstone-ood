#!/bin/bash

# custom config
DATA=../../DATA
TRAINER=NegPrompt
DATASET=imagenet_openood

CFG=$1  # config file
NCTX=$2
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
NEGA_CTX=$4

for SEED in 1
do
    # bash script for negprompt
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.NEGPROMPT.NEGA_CTX ${NEGA_CTX} \
        DATASET.NUM_SHOTS ${SHOTS}
        
    fi
done