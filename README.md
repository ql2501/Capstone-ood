# Capstone-ood
This group project is part of the Columbia University Data Science Capstone course, aimed at developing solutions for out-of-distribution (OOD) challenges. Our primary focus is on deploying and enhancing CoOp with NegPrompt methods, following the frameworks and methodologies outlined in the research papers available in [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134) and [Learning Transferable Negative Prompts for Out-of-Distribution Detection](https://arxiv.org/abs/2404.03248).

### To run CoOp code on Imagenet tiny images:
Set up Dassl env

```bash
cd CoOp_works/Dassl.pytorch

# Create a conda environment
conda create -y -n dassl python=3.8

# Activate the environment
conda activate dassl

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch  (old instruction from dassl's readme)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# use your suitable cudatookit version (11.8, 12.1, 12.4)

pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

If there's no error, go to CoOp directory

```bash
cd ..
cd CoOp
pip install -r requirements.txt
```

#####################################  
Then, choose one of the following:  
If training on CoOp, remove previous results for the new training on imagenet

```bash
rm -rf output/imagenet/CoOp  
```

If training on NegPrompt, remove previous results for the new training on imagenet

```bash
rm -rf output/imagenet/NegPrompt  
```

#####################################  
Then, choose one of the following:  
If training on CoOp, call bash scripts to train CoOp on ImagetNet

```bash
bash scripts/coop/main.sh imagenet rn50_ep50 end 16 1 False
```

If training on NegPrompt, call bash scripts to train NegPrompt on ImagetNet

```bash
bash scripts/negprompt/main.sh imagenet vit_b16_ep50 end 16 1 False
```
  
If run successfully, the output models and logs should be in the path above.  
  
Finally, deactivate the environment

```bash
conda deactivate
```

### To run in Windows PowerShell (may also be invoked in cmd)

Open PowerShell, run:

```powershell
conda init powershell
```

to initialize conda in PowerShell. Then, restart and run:

For Coop:

```powershell
conda activate dassl
cd CoOp_works/CoOp
powershell Remove-Item -Path "output/imagenet/CoOp" -Recurse -Force
powershell -File scripts/coop/main.ps1 imagenet rn50_ep50 end 16 1 False
conda deactivate
```

To run in one line:

```powershell
conda activate dassl && cd CoOp_works/CoOp && powershell Remove-Item -Path "output/imagenet/CoOp" -Recurse -Force && powershell -File scripts/coop/main.ps1 imagenet rn50_ep50 end 16 1 False && conda deactivate
```

or to rerun in one line:

```powershell
powershell Remove-Item -Path "output/imagenet/CoOp" -Recurse -Force && powershell -File scripts/coop/main.ps1 imagenet rn50_ep50 end 16 1 False
```

For NegPrompt:

```powershell
conda activate dassl
cd CoOp_works/CoOp
powershell Remove-Item -Path "output/imagenet/NegPrompt" -Recurse -Force
powershell -File scripts/negprompt/main.ps1 imagenet vit_b16_ep50 end 16 1 False
conda deactivate
```

To run in one line:

```powershell
conda activate dassl && cd CoOp_works/CoOp && powershell Remove-Item -Path "output/imagenet/NegPrompt" -Recurse -Force && powershell -File scripts/negprompt/main.ps1 imagenet vit_b16_ep50 end 16 1 False && conda deactivate
```

or to rerun in one line:

```powershell
powershell Remove-Item -Path "output/imagenet/NegPrompt" -Recurse -Force && powershell -File scripts/negprompt/main.ps1 imagenet vit_b16_ep50 end 16 1 False
```
