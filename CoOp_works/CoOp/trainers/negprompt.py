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
import torch.distributions as dist

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from datasets.classname import *

_tokenizer = _Tokenizer()
# Ensure the device is set globally, 解决CPU、CUDA使用问题 Yuhao add
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        print(prompts.shape)
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
            prompt = clip.tokenize(ctx_init).to(device) # move to correct device
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype).to(device)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            ctx_vectors = ctx_vectors.view(1, ctx_vectors.shape[0], ctx_vectors.shape[1]).to(device) # class_posi, ctx, vector
            ctx_vectors = ctx_vectors.repeat(1+n_nega_ctx, 1, 1)    # expand in first dimension (not batch)
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg['CSC']>0:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, 1+n_nega_ctx, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(1+n_nega_ctx, n_ctx, ctx_dim, dtype=dtype)

            ctx_vectors = ctx_vectors.to(device)
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
        ctx_positive = ctx_positive.to(device)
        ctx_negative = ctx_negative.to(device)
        self.ctx_positive = nn.Parameter(ctx_positive)  # to be optimized
        if ctx_negative.shape[0] == 0:
            ctx_negative = torch.empty(0, dtype=dtype).to(device)
        self.ctx_negative = nn.Parameter(ctx_negative)  # to be optimized
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        positive_prompts = [prompt_prefix + " " +  name   for name in classnames]
        negative_prompts = [prompt_prefix + " " + name  for name in classnames]     # same as positive prompts
        # breakpoint()
        positive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in positive_prompts]).to(device)
        negative_tokenized_prompts = torch.cat([clip.tokenize(p) for p in negative_prompts]).to(device)
        # tokenized_prompts:
        # tensor([ <start>    a     photo   of   a  positive [classname] . <end>
                # [49406,   320,  1125,   539,   320,  4844,  1929,   269, 49407, 0 ...,0],
                # [49406,   320,  1125,   539,   320,  4844,  2368,   269, 49407, 0 ...,0],
                # [49406,   320,  1125,   539,   320,  4844,  4558,   269, 49407, 0 ...,0],
                # [49406,   320,  1125,   539,   320,  4844,  6531,   269, 49407, 0 ...,0]])
        with torch.no_grad():
            positive_embedding = clip_model.token_embedding(positive_tokenized_prompts).type(dtype).to(device)
            negative_embedding = clip_model.token_embedding(negative_tokenized_prompts).type(dtype).to(device)
        
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

    # Returns the prompt vectors for the positive class names.
    def forward_positive(self):
        print("Reached forward_positive in NegaPromptLearner")
        ctx_positive = self.ctx_positive
        if ctx_positive.dim() == 3:
            ctx = ctx_positive.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        else:
            ctx = ctx_positive
        prefix = self.positive_token_prefix
        suffix = self.positive_token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls,1+n_neg, 1, dim)
                ctx,     # (n_cls,1+n_neg, n_ctx, dim)
                suffix,  # (n_cls,1+n_neg, *, dim)
            ],
            dim = 2,
        )
        return prompts
    # Returns the prompt vectors for the negative class names only.
    def forward_negative(self):
        print("Reached forward_negative in NegaPromptLearner")
        ctx_negative = self.ctx_negative.to(device)
        if ctx_negative.dim() == 3:
            ctx = ctx_negative.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        else:
            ctx = ctx_negative
        prefix = self.negative_token_prefix.to(device)
        suffix = self.negative_token_suffix.to(device)
        prompts = torch.cat(
            [
                prefix,  # (n_cls,1+n_neg, 1, dim)
                ctx,     # (n_cls,1+n_neg, n_ctx, dim)
                suffix,  # (n_cls,1+n_neg, *, dim)
            ],
            dim = 2,
        )
        return prompts

    # Update the positive context vectors by given ctx_posi, and generate negative context vectors.
    # "ctx_posi" is a torch.Tensor
    def update_ctx_positive(self, ctx_posi):
        noise_range = 1e-5
        noise_dist = dist.Uniform(low=-noise_range, high=noise_range, )
        if ctx_posi.dim() == 2: 
            ctx_posi = ctx_posi.unsqueeze(0)
        if self.csc == 1:
            ctx_negative_repeated = ctx_posi.repeat(1, self.n_nega_ctx, 1, 1)
        else:
            ctx_negative_repeated = ctx_posi.repeat(self.n_nega_ctx, 1, 1)
        ctx_negative = ctx_negative_repeated + noise_dist.sample(ctx_negative_repeated.shape).to(self.ctx_negative.device)
        ctx_negative = ctx_negative.half()
        self.ctx_positive = nn.Parameter(ctx_posi, requires_grad=False)
        self.ctx_negative = nn.Parameter(ctx_negative, requires_grad=True)
        print(f"After update, the shape of ctx_positive is {self.ctx_positive.shape}")
    
    def update_ctx_negative(self, ctx_nega):
        pass

    def freeze_ctx_positive(self):
        pass

    def get_ctx_positive(self):
        return self.ctx_positive

'''
Not exist in NegPrompt, but needed in order to implement a complete CoOp trainer
'''
# 中间层
class NegPromptCustomCLIP(nn.Module):
    # Qi's modification: 
    # this __init__ should follow the one in NegPromptClip instead of the one in CoOp
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = NegaPromptLearner(cfg, classnames, clip_model).to(device)
        self.n_nega_ctx = cfg.TRAINER.NEGPROMPT.NEGA_CTX
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual.to(device)
        self.text_encoder = NegaTextEncoder(clip_model).to(device)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames
        self.positive_text_features = None
        self.clip_model = clip_model
        self.cfg = cfg

    # Qi: 
    # In our case, we only train negative prompts, corresponding to stage 3 in NegPrompt
    def forward(self, image):
        return self.forward_negative(image)
    
    # Only learn the negative prompts
    # return shape:
    # logits: [batch_size, nclass * 1+n_nega_ctx]
    # text_features: [nclass * 1+n_nega_ctx, 512]
    def forward_negative(self, image): 
        print("Reached forward_negative in NegPromptCustomCLIP")
        image_features = self.image_encoder(image.to(device).type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        negative_prompts = self.prompt_learner.forward_negative()    # use negative prompts only
        negative_tokenized_prompts = self.prompt_learner.negative_tokenized_prompts
        negative_text_features = self.text_encoder(negative_prompts, negative_tokenized_prompts) #(1000*n_nega_ctx) * 512)
        positive_text_features = self.positive_text_features # 1000*512, fixed
        #fusion the text_features that positive, negative, positive, negative, ...
        positive_text_features = positive_text_features.view(positive_text_features.shape[0], 1, -1)
        negative_text_features = negative_text_features.view(positive_text_features.shape[0], self.n_nega_ctx, -1)  # 1000 * n_nega_ctx * 512

        # here we concatenate the positive and negative text features
        text_features = torch.cat([positive_text_features, negative_text_features], dim=1)  
        text_features = text_features.view(text_features.shape[0]*text_features.shape[1], -1)   # 1000*(1+n_nega_ctx) * 512
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # shape: 1000*(1+n_nega_ctx) * 512
        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t())
        return logits, text_features

    # Use ctx_posi to update the positive context vectors, and generate negative context vectors.
    # Then get the positive text features by CLIP text encoder into positive_text_features.
    def get_ctx_posi(self, ctx_posi):
        self.prompt_learner.update_ctx_positive(ctx_posi)
        # get positive_text_features
        prompts = self.prompt_learner.forward_positive() # Returns the prompt vectors for the positive class names.
        tokenized_prompts = self.prompt_learner.positive_tokenized_prompts
        self.positive_text_features = self.text_encoder(prompts, tokenized_prompts) # get text embedding for positive prompts by CLIP transformer

'''
    Migrate NegaPromptCLIP to the following NegPrompt class
''' 
# 这个是最上面的一层，通过中间层包在NegaPromptLearner和NegaTextEncoder上面. 
@TRAINER_REGISTRY.register()
class NegPrompt(TrainerX):
    # Override build_model in line 324 from CoOp_works\Dassl.pytorch\dassl\engine\trainer.py
    # called in SimpleTrainer init
    def build_model(self):
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.build_model")
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg).to(device)
        print(f"Successfully loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        print('-' * 80)

        if cfg.TRAINER.NEGPROMPT.PREC == "fp32" or cfg.TRAINER.NEGPROMPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        
        # Re-implemented NegPromt's CustomClip()
        print("Building NegPrompt's custom CLIP")
        self.model = NegPromptCustomCLIP(cfg, classnames, clip_model).to(device)
        print(f"Successfully building NegPrompt's custom CLIP")
        print('-' * 80)

        # Migrate parameter freezing logic in NegPrompt, freeze positive prompts as well
        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name or "ctx_positive" in name:
                param.requires_grad_(False)
            else: 
                print(f"Remaining active gradient in {name}, paramter shape {param.shape}")\

        # if need to initialize weights 
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        # self.model.to(self.device)
        # Only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        print("Finished building model NegPrompt")

    # Qi: choose to load positive prompts here 
    def before_train(self):
        # still need to call before_train in Dassl
        super().before_train()
        
        # load
        # NOTE: this path目前是写死的，之后看看要不要变成一个参数存在cfg里
        model_positive_path = "output/imagenet/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50"
        print(f"Start loading positive prompts from model path {model_positive_path}")
        saved_model = torch.load(model_positive_path, map_location=torch.device(device))
        print("Loaded positive prompts: ")
        print(saved_model['state_dict']['ctx'])
        self.model.get_ctx_posi(saved_model['state_dict']['ctx'])
        print('Positive prompt from Pretrained CoOp loaded')
        del saved_model

    # dummy method 
    def load_model(self, directory, epoch=None): 
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.load_model")

    # dummy method 
    def test(self): 
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.test")

    # 之后train应该会用到
    def forward_backward(self, batch):
        raise NotImplementedError
    
    # 之后train应该会用到
    def parse_batch_train(self, batch):
        raise NotImplementedError
    
    # Qi: 
    # TODO: need to check if we really need to override this method
    def run_epoch(self):
        pass
    
    # TODO
    def after_epoch(self):
        pass

    # Qi: dummy train for debugging (我不觉得需要override这个)
    def train(self): 
        print("Calling train")
        self.before_train()
        print("Before train done")
        # self.before_epoch()
        # self.run_epoch()
        # self.after_epoch()
        # self.after_train()
