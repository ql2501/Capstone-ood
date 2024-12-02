#!/bin/bash

# custom config
DATA=../../DATA
TRAINER=NegPrompt
DATASET=imagenet_openood

CFG=$1  # config file
SHOTS=$2  # number of shots (1, 2, 4, 8, 16)


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
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done