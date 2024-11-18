# main model code for negprompt

# Dongqi 的一些建议：
    # 目前的目标是用CoOp的trainer框架复现NegPrompt的功能
    # 我感觉最好的approach是把CoOp build_model中的每一步复现
    # 然后遇到什么卡住就迁移什么，这样至少有一个着手点
    # 我把目前每个类的迁移进展写在了定义前的literal string里，大家可以update
import os.path as osp
import re

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

from datasets.classname import *

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

'''
目前看来init这个类需要的参数都是backbone中定义的clip_model(由load_clip_to_cpu返回)中自带的参数
暂时没有要对齐的东西
someone check this please
'''
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


'''
暂时只migrate过来了init function to unblock build_model
'''
# This is corresponding to PromptLearner In Coop
class NegaPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.NEGPROMPT.N_CTX
        if 'ImageNet' in cfg.DATASET:
            print(f"cfg all dataset looks like:{cfg.DATASET}")
            ctx_init = 'a photo of a "{}"'
        else:
            ctx_init = classname_dic[cfg.DATASET.NAME]["templates"][0]
        if cfg.TRAINER.NEGPROMPT.CSC>0: # In NegaPrompt, this defaults to 0
            ctx_init = None
        n_nega_ctx = cfg.TRAINER.NEGPROMPT.NEGA_CTX    # negative context
        self.n_nega_ctx = n_nega_ctx    # number of negative context
        self.csc = cfg.TRAINER.NEGPROMPT.CSC
        self.cfg = cfg
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("\"{}\"", "")
            ctx_init = ctx_init.replace(".", "")
            ctx_init = ctx_init.replace("_", " ")
            words = re.findall(r'\b\w+\b', ctx_init)
            n_ctx = len(words)
            prompt = clip.tokenize(ctx_init)
            prompt = prompt.cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype) 
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            ctx_vectors = ctx_vectors.view(1, ctx_vectors.shape[0], ctx_vectors.shape[1]) # class_posi, ctx, vector
            ctx_vectors = ctx_vectors.repeat(1+n_nega_ctx, 1, 1)    # expand in first dimension (not batch)
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg['CSC']>0:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, 1+n_nega_ctx, n_ctx, ctx_dim, dtype=dtype).cuda()
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(1+n_nega_ctx, n_ctx, ctx_dim, dtype=dtype).cuda()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # split the context vector
        if ctx_vectors.dim() == 3:
            ctx_positive = ctx_vectors[0:1, :, :]
            ctx_negative = ctx_vectors[1:, :, :]
        else:
            ctx_positive = ctx_vectors[:, 0:1, :, :]
            ctx_negative = ctx_vectors[:, 1:, :, :]
        self.ctx_positive = nn.Parameter(ctx_positive)  # to be optimized
        if ctx_negative.shape[0] == 0:
            ctx_negative = torch.empty(0, dtype=dtype).cuda()
        self.ctx_negative = nn.Parameter(ctx_negative)  # to be optimized
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        positive_prompts = [prompt_prefix + " " +  name   for name in classnames]
        negative_prompts = [prompt_prefix + " " + name  for name in classnames]     # same as positive prompts
            
        positive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in positive_prompts]).cuda()
        negative_tokenized_prompts = torch.cat([clip.tokenize(p) for p in negative_prompts]).cuda()
        # tokenized_prompts:
        # tensor([ <start>    a     photo   of   a  positive [classname] . <end>
                # [49406,   320,  1125,   539,   320,  4844,  1929,   269, 49407, 0 ...,0],
                # [49406,   320,  1125,   539,   320,  4844,  2368,   269, 49407, 0 ...,0],
                # [49406,   320,  1125,   539,   320,  4844,  4558,   269, 49407, 0 ...,0],
                # [49406,   320,  1125,   539,   320,  4844,  6531,   269, 49407, 0 ...,0]])
        with torch.no_grad():
            positive_embedding = clip_model.token_embedding(positive_tokenized_prompts).type(dtype)
            negative_embedding = clip_model.token_embedding(negative_tokenized_prompts).type(dtype)
        
        # get embeddings
        # squeeze the dimension 1
        positive_embedding = positive_embedding.view(positive_embedding.shape[0], 1, positive_embedding.shape[1], positive_embedding.shape[2])
        negative_embedding = negative_embedding.view(negative_embedding.shape[0], 1, negative_embedding.shape[1], negative_embedding.shape[2])
        negative_embedding = negative_embedding.repeat(1, n_nega_ctx, 1, 1)
        embedding = torch.cat([positive_embedding, negative_embedding], dim=1)
        positive_tokenized_prompts = positive_tokenized_prompts.view(positive_tokenized_prompts.shape[0], 1, positive_tokenized_prompts.shape[1])
        negative_tokenized_prompts = negative_tokenized_prompts.view(negative_tokenized_prompts.shape[0], 1, negative_tokenized_prompts.shape[1])
        negative_tokenized_prompts = negative_tokenized_prompts.repeat(1, n_nega_ctx, 1)
        tokenized_prompts = torch.cat([positive_tokenized_prompts, negative_tokenized_prompts], dim=1)
        tokenized_prompts = tokenized_prompts.view(tokenized_prompts.shape[0]*tokenized_prompts.shape[1], -1)
        self.register_buffer("token_prefix", embedding[:, :, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, :, 1 + n_ctx :, :])  # positive prompt CLS, EOS
        if cfg['stage'] >= 2:
            self.register_buffer("positive_token_prefix", embedding[:, :1, :1, :])  # SOS
            self.register_buffer("positive_token_suffix", embedding[:, :1, 1 + n_ctx :, :])  # positive prompt CLS, EOS
            self.register_buffer("negative_token_prefix", embedding[:, 1:, :1, :])  # SOS
            self.register_buffer("negative_token_suffix", embedding[:, 1:, 1 + n_ctx :, :])
            self.positive_tokenized_prompts = positive_tokenized_prompts.view(positive_tokenized_prompts.shape[0]*positive_tokenized_prompts.shape[1], -1)
            self.negative_tokenized_prompts = negative_tokenized_prompts.view(negative_tokenized_prompts.shape[0]*negative_tokenized_prompts.shape[1], -1)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

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

'''
foward暂时没有migrate
'''
# 中间层
# Not exist in NegPrompt, but needed in order to implement a complete CoOp trainer
class NegPromptCustomCLIP(nn.Module):
    # Qi's modification: 
    # this __init__ should follow the one in NegPromptClip instead of the one in CoOp
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = NegaPromptLearner(cfg, classnames, clip_model).cuda()
        self.n_nega_ctx = cfg['NEGA_CTX']
        self.stage = cfg['stage']
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = NegaTextEncoder(clip_model).cuda()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames
        self.positive_text_features = None
        self.clip_model = clip_model
        self.cfg = cfg

    def forward(self, image):
        pass

# TODO: migrate NegaPromptCLIP to the following NegPrompt class
# 这个是最上面的一层，通过中间层包在NegaPromptLearner和NegaTextEncoder上面. 
@TRAINER_REGISTRY.register()
class NegPrompt(TrainerX):
    # dummy method 
    def load_model(self, directory, epoch=None): 
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.load_model")

    # dummy method 
    def test(self): 
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.test")

    # CoOp_works\Dassl.pytorch\dassl\engine\trainer.py 第324行，会在initialize SimpleTrainer的时候call build_model
    # 所以这里要override一下
    def build_model(self):
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.build_model")
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        print(f"Successfully loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        print('-' * 80)

        if cfg.TRAINER.NEGPROMPT.PREC == "fp32" or cfg.TRAINER.NEGPROMPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        
        # TODO: 复现NegPromt版本的CustomClip()
        print("TODO: Building NegPrompt's custom CLIP")
        self.model = NegPromptCustomCLIP(cfg, classnames, clip_model)
        print(f"Successfully building NegPrompt's custom CLIP")
        print('-' * 80)

        # TODO: Migrate parameter freezing logic in NegPrompt, freeze positive prompts as well
        print("Migrating from CoOp, turning off gradients in both the image and the text encoder")
        # for name, param in self.model.named_parameters():
        #     if "prompt_learner" not in name:
        #         param.requires_grad_(False)


    # 之后train应该会用到
    def forward_backward(self, batch):
        raise NotImplementedError
    
    # 之后train应该会用到
    def parse_batch_train(self, batch):
        raise NotImplementedError
    
    # override SimpleTrainer的train()
    def train(self): 
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.train")