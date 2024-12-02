# custom config
$DATA = "../../DATA"
$TRAINER = "NegPrompt"

$DATASET = $args[0]
$CFG = $args[1]  # config file

# Specify the model directory and epoch
$MODEL_DIR = ".\output\imagenet\NegPrompt\try_config\seed1"
$LOAD_EPOCH = 50

$SEEDS = 1..1

foreach ($SEED in $SEEDS) {
    # PowerShell script for negprompt
    # $DIR = "output/$DATASET/$TRAINER/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}"
    $DIR = "output/$DATASET/$TRAINER/try_config/seed${SEED}" # dummy directory for config debugging 
    python train.py `
    --root $DATA `
    --seed $SEED `
    --trainer $TRAINER `
    --dataset-config-file "configs/datasets/$DATASET.yaml" `
    --config-file "configs/trainers/$TRAINER/$CFG.yaml" `
    --output-dir $DIR `
    --eval-only `
    --no-train `
    --model-dir $MODEL_DIR `
    --load-epoch $LOAD_EPOCH
}
# conda activate dassl
# cd CoOp_works/CoOp
# powershell -File scripts/negprompt/eval.ps1 imagenet vit_b16_ep50 end 16 1 False
# powershell -File scripts/negprompt/eval.ps1 texture vit_b16_ep50 end 16 1 False