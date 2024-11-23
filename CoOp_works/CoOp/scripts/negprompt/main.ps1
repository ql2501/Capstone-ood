# custom config
$DATA = "../../DATA"
$TRAINER = "NegPrompt"

$DATASET = $args[0]
$CFG = $args[1]  # config file
$CTP = $args[2]  # class token position (end or middle)
$NCTX = $args[3]  # number of context tokens
$SHOTS = $args[4]  # number of shots (1, 2, 4, 8, 16)
$CSC = $args[5]  # class-specific context (False or True)

$SEEDS = 1..1

foreach ($SEED in $SEEDS) {
    # PowerShell script for negprompt
    # $DIR = "output/$DATASET/$TRAINER/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}"
    $DIR = "output/$DATASET/$TRAINER/try_config/seed${SEED}" # dummy directory for config debugging 
    if (Test-Path $DIR) {
        Write-Output "Oops! The results exist at $DIR (so skip this job)"
    } else {
        python train.py `
        --root $DATA `
        --seed $SEED `
        --trainer $TRAINER `
        --dataset-config-file "configs/datasets/$DATASET.yaml" `
        --config-file "configs/trainers/$TRAINER/$CFG.yaml" `
        --output-dir $DIR
        DATASET.NUM_SHOTS $SHOTS

        # qi_liu: don't think this extra opts is necessary

        # TRAINER.COOP.N_CTX $NCTX `
        # TRAINER.COOP.CSC $CSC `
        # TRAINER.COOP.CLASS_TOKEN_POSITION $CTP `
    }
}
# conda activate dassl
# cd CoOp_works/CoOp
# powershell Remove-Item -Path "output/imagenet/NegPrompt" -Recurse -Force
# powershell -File scripts/negprompt/main.ps1 imagenet vit_b16_ep50 end 16 1 False

# conda activate dassl && cd CoOp_works/CoOp && powershell Remove-Item -Path "output/imagenet/NegPrompt" -Recurse -Force && powershell -File scripts/negprompt/main.ps1 imagenet vit_b16_ep50 end 16 1 False
# powershell Remove-Item -Path "output/imagenet/NegPrompt" -Recurse -Force && powershell -File scripts/negprompt/main.ps1 imagenet vit_b16_ep50 end 16 1 False