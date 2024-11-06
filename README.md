# Capstone-ood
This group project is part of the Columbia University Data Science Capstone course, aimed at developing solutions for out-of-distribution (OOD) challenges. Our primary focus is on deploying and enhancing CoOp with NegPrompt methods, following the frameworks and methodologies outlined in the research papers available in [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134) and [Learning Transferable Negative Prompts for Out-of-Distribution Detection](https://arxiv.org/abs/2404.03248).

### To run CoOp code on Imagenet tiny images:
```bash
cd /CoOp_works/CoOp
```
```bash
conda activate dassl
```
```bash
bash scripts/coop/main.sh imagenet rn50_ep50 end 16 1 False
```
```bash
conda deactivate
