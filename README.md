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
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
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
Remove previous results for the new training on imagenet
```bash
rm -rf output/imagenet
```
Call bash scripts to train CoOp on ImagetNet
```bash
bash scripts/coop/main.sh imagenet rn50_ep50 end 16 1 False
```
Finally, deactivate the environment
```bash
conda deactivate
