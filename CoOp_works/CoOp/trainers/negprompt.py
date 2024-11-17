# main model code for negprompt
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# Migrating helper functions from CoOp/trainers/coop.py to here
# Try to keep NegaPrompt trainer structure consistant with CoOp trainer structure
def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

# NegaTextEncoder migrated from NegaPrompt, this should be corresponding to the TextEncoder from CoOp
class NegaTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.transformer.eval()
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        if(hasattr(clip_model, 'attn_mask')):
            self.attn_mask = clip_model.attn_mask
        else:
            self.attn_mask = None
        # print('attn_mask is ', self.attn_mask)
    
    def forward(self, prompts, tokenized_prompts):
        '''
        Encodes the given text prompts using the CLIP transformer.
        '''
        if len(prompts.shape) == 4:
            prompts = torch.flatten(prompts, start_dim=0, end_dim=1)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND (n_class*(1+n_neg)) * n_ctx * dim 
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(x.device)
            x = self.transformer(x, self.attn_mask)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


# This is corresponding to PromptLearner In Coop
class NegaPromptLearner(nn.Module):
    def forward(self):
        pass

    # TODO: further check if foward_positive is needed. If not discard it
    def foward_positive(self):
        pass

    def foward_negative(self):
        pass

    def update_ctx_positive(self, ctx_posi):
        pass
    
    def update_ctx_negative(self, ctx_nega):
        pass

    def freeze_ctx_positive(self):
        pass

    def get_ctx_positive(self):
        pass


# TODO: migrate NegaPromptCLIP to the following NegPrompt class
# 这个是最上面的一层，包在NegaPromptLearner和NegaTextEncoder上面. 
@TRAINER_REGISTRY.register()
class NegPrompt(TrainerX):
    """
        Dummy class for NegPrompt for config debugging
    """

    # dummy method 
    def load_model(self, directory, epoch=None): 
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.load_model")

    # dummy method 
    def test(self): 
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.test")

    # dummy method
    # CoOp_works\Dassl.pytorch\dassl\engine\trainer.py 第324行，会在initialize SimpleTrainer的时候call build_model
    # 所以这里要override一下
    def build_model(self):
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.build_model")

    # 之后train应该会用到
    def forward_backward(self, batch):
        raise NotImplementedError
    
    # 之后train应该会用到
    def parse_batch_train(self, batch):
        raise NotImplementedError
    
    # override SimpleTrainer的train()
    def train(self): 
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.train")