#!/bin/bash

# Custom config
DATA=../../DATA
TRAINER=NegPrompt

DATASET=$1    # First argument
CFG=$2        # Second argument (config file)

# Specify the model directory and epoch
MODEL_DIR=$3
LOAD_EPOCH=$4


# Iterate over seeds
for SEED in 1
do
    # DIR="output/$DATASET/$TRAINER/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}"
    DIR="output/$DATASET/$TRAINER/eval/model_dir${MODEL_DIR}/load_epoch${LOAD_EPOCH}/seed${SEED}" # Dummy directory for config debugging

    # Execute the Python script
    python train.py \
        --root "$DATA" \
        --seed "$SEED" \
        --trainer "$TRAINER" \
        --dataset-config-file "configs/datasets/${DATASET}.yaml" \
        --config-file "configs/trainers/$TRAINER/${CFG}.yaml" \
        --output-dir "$DIR" \
        --eval-only \
        --no-train \
        --model-dir "$MODEL_DIR" \
        --load-epoch "$LOAD_EPOCH"
done