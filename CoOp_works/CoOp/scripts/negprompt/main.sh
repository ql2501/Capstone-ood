#!/bin/bash

# custom config
DATA=../../DATA
TRAINER=NegPrompt

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)


for SEED in 1 2 3
do
    # bash script for negprompt
    # DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    DIR=output/${DATASET}/${TRAINER}/try_config/seed${SEED} # dummy directory for config debugging 
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

        # qi_liu: don't think this extra opts is necessary

        # TRAINER.COOP.N_CTX ${NCTX} \
        # TRAINER.COOP.CSC ${CSC} \
        # TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        # DATASET.NUM_SHOTS ${SHOTS}
    fi
done