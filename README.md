# Capstone-ood
This group project is part of the Columbia University Data Science Capstone course, aimed at developing solutions for out-of-distribution (OOD) challenges. Our primary focus is on deploying and enhancing CoOp with NegPrompt methods, following the frameworks and methodologies outlined in the research papers available in [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134) and [Learning Transferable Negative Prompts for Out-of-Distribution Detection](https://arxiv.org/abs/2404.03248).

### To run CoOp code on Imagenet tiny images:
Go to CoOp directory
```bash
cd /CoOp_works/CoOp
```
Activate the dassl environment in order to run CoOp
```bash
conda activate dassl
```
Call bash scripts to train CoOp on ImagetNet
```bash
bash scripts/coop/main.sh imagenet rn50_ep50 end 16 1 False
```
Finally, deactivate the environment
```bash
conda deactivate
