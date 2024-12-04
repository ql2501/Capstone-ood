'''
    main model code for negprompt
'''

import os.path as osp
import re

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from datasets.classname import *
from trainers.negprompt_utils import compute_oscr, compute_fpr, metric_ood

from tqdm import tqdm
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import os

_tokenizer = _Tokenizer()
# ensure the device is set globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for debugging purpose
DEBUG = False    

def load_clip_to_cpu(cfg):
    '''
    Migrating helper functions from CoOp/trainers/coop.py to here
    Try to keep NegaPrompt trainer structure consistant with CoOp trainer structure
    '''
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
NegaTextEncoder migrated from NegaPrompt, this should be corresponding to the TextEncoder from CoOp
'''
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
This is corresponding to PromptLearner In Coop
'''
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

        positive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in positive_prompts]).to(device)
        negative_tokenized_prompts = torch.cat([clip.tokenize(p) for p in negative_prompts]).to(device)
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

    def forward(self, modify_to_ori = None):
        '''
        Returns the prompt vectors that contains both positive and negative prompts.
        Called when testing or validating the model.
        '''
        # modify_to_ori is a dic that transform the modified labels to original ones. This maybe used when sample 10 classes from 1k classes.
        ctx_positive = self.ctx_positive
        ctx_negative = self.ctx_negative
        # make ctx_negative[0,0,:] to ctx_negative
        if ctx_negative.shape[0] == 0:
            if ctx_positive.dim() == 3:
                ctx = ctx_positive.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            else:
                ctx = ctx_positive
        else:   
            if ctx_positive.dim() == 3:
                diff = ctx_positive.shape[1] - ctx_negative.shape[1]
                additional_rows = torch.zeros((ctx_negative.shape[0], diff, ctx_negative.shape[2]) ).cuda()
                additional_rows = additional_rows.to(ctx_negative.dtype)
                ctx_negative = torch.cat([additional_rows, ctx_negative], dim=1)
                ctx = torch.cat([ctx_positive, ctx_negative], dim=0)
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)   # shape: (n_cls, 1+n_neg, n_ctx, dim)
            else:
                ctx = torch.cat([ctx_positive, ctx_negative], dim=1)    # train them together
        prefix = self.token_prefix
        suffix = self.token_suffix

        if modify_to_ori is not None:
            # modify_to_ori is a dic that transform the modified labels to original ones
            # This maybe used when sample 10 classes from 1k classes.
            ori_labels = list(modify_to_ori.values())
            ctx = ctx[ori_labels]   # only keep the classes needed TODO: check if this is correct
            prefix = prefix[ori_labels]
            suffix = suffix[ori_labels]
        prompts = torch.cat(
            [
                prefix,  # (n_cls,1+n_neg, 1, dim)
                ctx,     # (n_cls,1+n_neg, n_ctx, dim)
                suffix,  # (n_cls,1+n_neg, *, dim)
            ],
            dim = 2,
        )
        
        return prompts

    def forward_positive(self):
        '''
        Returns the prompt vectors for the positive class names.
        '''
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
    
    def forward_negative(self):
        '''
        Returns the prompt vectors for the negative class names only.
        '''
        if DEBUG: print("Reached forward_negative in NegaPromptLearner")
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

    def update_ctx_positive(self, ctx_posi):
        '''
        Update the positive context vectors by given ctx_posi, and generate negative context vectors.
        NOTE: "ctx_posi" is a torch.Tensor
        '''
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
        '''
        Set the negative context vectors to ctx_nega.
        '''
        self.ctx_negative = nn.Parameter(ctx_nega, requires_grad=False)

    def freeze_ctx_positive(self):
        '''
        Freeze the positive context vectors to self.ctx_positive.
        '''
        self.ctx_positive = nn.Parameter(self.ctx_positive, requires_grad=False)

    def get_ctx_positive(self):
        return self.ctx_positive

'''
Intermediate layer: not exist in NegPrompt, but needed in order to implement a complete CoOp trainer
'''
class NegPromptCustomCLIP(nn.Module): 
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

    def forward(self, image):
        '''
        In our case, we only train negative prompts, corresponding to stage 3 in NegPrompt
        '''
        return self.forward_negative(image)
    
    def forward_negative(self, image): 
        '''
        Only learn the negative prompts
        image: [batch_size, 3, 224, 224]
        return shape:
        logits: [batch_size, nclass * 1+n_nega_ctx]
        text_features: [nclass * 1+n_nega_ctx, 512]
        '''
        if DEBUG: print("Reached forward_negative in NegPromptCustomCLIP")
        image_features = self.image_encoder(image.to(device).type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        negative_prompts = self.prompt_learner.forward_negative()    # use negative prompts only
        negative_tokenized_prompts = self.prompt_learner.negative_tokenized_prompts
        negative_text_features = self.text_encoder(negative_prompts, negative_tokenized_prompts) #(1000*n_nega_ctx) * 512)
        positive_text_features = self.positive_text_features # 1000*512, fixed
        # fusion the text_features that positive, negative, positive, negative, ...
        positive_text_features = positive_text_features.view(positive_text_features.shape[0], 1, -1)
        negative_text_features = negative_text_features.view(positive_text_features.shape[0], self.n_nega_ctx, -1)  # 1000 * n_nega_ctx * 512

        # here we concatenate the positive and negative text features
        text_features = torch.cat([positive_text_features, negative_text_features], dim=1)  
        text_features = text_features.view(text_features.shape[0]*text_features.shape[1], -1)   # 1000*(1+n_nega_ctx) * 512
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # shape: 1000*(1+n_nega_ctx) * 512
        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t())
        return logits, text_features

    def get_ctx_posi(self, ctx_posi):
        '''
        Use ctx_posi to update the positive context vectors, and generate negative context vectors.
        Then get the positive text features by CLIP text encoder into positive_text_features.
        '''
        self.prompt_learner.update_ctx_positive(ctx_posi)
        # get positive_text_features
        prompts = self.prompt_learner.forward_positive() # Returns the prompt vectors for the positive class names.
        tokenized_prompts = self.prompt_learner.positive_tokenized_prompts
        self.positive_text_features = self.text_encoder(prompts, tokenized_prompts) # get text embedding for positive prompts by CLIP transformer

    def forward_test(self, image, text_features=None):
        '''
        The forward method for testing, need input trained text_features
        '''
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t())
        return logits, text_features

'''
Migrate NegaPromptCLIP to the following NegPrompt class
This is the topmost layer, wrapping around the NegaPromptLearner and NegaTextEncoder through the intermediate layer.
'''  
@TRAINER_REGISTRY.register()
class NegPrompt(TrainerX):
    def build_model(self):
        '''
        Override build_model in line 324 from CoOp_works\Dassl.pytorch\dassl\engine\trainer.py
        called in SimpleTrainer init
        '''
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
        self.n_cls = len(classnames)    # just to debug
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
        
        # load positive prompts here: 
        # self.positive_text_features is updated
        # self.ctx_positive is updated
        # self.ctx_negative is updated
        # NOTE: This path is currently hardcoded. Consider whether to make it a parameter in the cfg later.
        model_positive_path = "output/imagenet/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50"
        print(f"Start loading positive prompts from model path: {model_positive_path}")
        saved_model = torch.load(model_positive_path, map_location=torch.device(device))
        print("Loaded positive prompts: ")
        print(saved_model['state_dict']['ctx'])
        self.model.get_ctx_posi(saved_model['state_dict']['ctx'])
        print('Positive prompt from Pretrained CoOp loaded')
        del saved_model

        # Only give prompt_learner to the optimizer
        # add only active params in self.model to optimizer
        params = []
        for name, param in self.model.prompt_learner.named_parameters():
            if param.requires_grad:
                print(f"Adding {name} to optimizer")
                params.append(param)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM, param_groups = params)
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

    def load_model(self, directory, epoch=None): 
        '''
        This is used if we want to load a model from a checkpoint and continue to train it
        '''
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.load_model")

        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    def get_ood_score(self, logits):
        '''
        logits shape: (batch_size, n_classes * (1+n_nega_ctx))
        Return: predictions, ood_score, logits_posi, logits_negas
        '''
        n_nega_ctx = self.model.n_nega_ctx
        softmax_logits = F.softmax(logits, dim=1)
        softmax_logits = softmax_logits.view(-1, int(softmax_logits.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx) # (batch_size, n_classes, 1+n_nega_ctx)
        logits = logits.view(-1, int(logits.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx) # (batch_size, n_classes, 1+n_nega_ctx)
        
        softmax_logits_posi = softmax_logits[:, :, 0]   # (batch_size, n_classes)
        softmax_logits_negas = softmax_logits[:, :, 1:]
        logits_posi = logits[:, :, 0]   # (batch_size, n_classes)
        logits_negas = logits[:, :, 1:]
        predictions = softmax_logits_posi.data.max(1)[1]    # (batch_size,)

        if self.cfg.TRAINER.NEGPROMPT.OPEN_SCORE == 'msp':  # maximum softmax probability, NOTE: now using this one
            ood_score = softmax_logits_posi #NOTE: use on cuda here for future processing #.data.cpu().numpy()  # (batch_size, n_classes)
        elif self.cfg.TRAINER.NEGPROMPT.OPEN_SCORE == 'maxlogit':
            ood_score = logits_posi.data.cpu().numpy()  # (batch_size, n_classes)
        elif self.cfg.TRAINER.NEGPROMPT.OPEN_SCORE == 'energy_oe':
            # calculate energy-based OOD score
            energy = torch.log(torch.sum(torch.exp(logits_posi), dim=1)).unsqueeze(1).cpu().numpy() # (batch_size, 1)
            ood_score = energy
        elif self.cfg.TRAINER.NEGPROMPT.OPEN_SCORE == 'nega':
            ood_score = softmax_logits_negas.data.max(2)[0].cpu().numpy()   # (batch_size, n_classes)
        elif self.cfg.TRAINER.NEGPROMPT.OPEN_SCORE == 'posi_nega':  # 计算正样本部分的 softmax 概率减去负样本部分的最大 softmax 概率作为 OOD 分数。
            nega_dis = torch.Tensor(softmax_logits_posi.shape[0]).cuda()    # (batch_size,)
            for i in range(softmax_logits_posi.shape[0]):
                nega_dis[i] = torch.max(softmax_logits_negas[i, predictions[i], :])
            nega_dis = nega_dis.view(-1, 1)
            nega_dis = nega_dis.repeat(1, softmax_logits_posi.shape[1])
            posi_minus_nega = softmax_logits_posi - nega_dis
            ood_score = posi_minus_nega.data.cpu().numpy()
        # NOTE: disable this two because we don't have radius function
        # elif self.cfg.TRAINER.NEGPROMPT.OPEN_SCORE == 'posi_minus_closest_radius':  # 计算正样本部分的 softmax 概率减去最近的半径值作为 OOD 分数。radius计算在models/models.py
        #     _, min_loc = torch.min(softmax_logits_negas, dim=2)
        #     index1 = torch.arange(min_loc.shape[1])
        #     index1 = index1.repeat(min_loc.shape[0]).cuda()
        #     index2 = min_loc.flatten().cuda()
        #     right_radius = radius[index1, index2].view(min_loc.shape[0], min_loc.shape[1]).cuda()
        #     posi_minus_radius = right_radius - softmax_logits_posi
        #     ood_score = posi_minus_radius.data.cpu().numpy()
        # elif self.cfg.TRAINER.NEGPROMPT.OPEN_SCORE == 'posi_radius':
        #     #right_radius(logits_posi.shape[0] * right_radius.shape[0]) is repeated by radius_mean\
        #     right_radius = radius_mean.expand((softmax_logits_posi.shape[0], -1)).cuda()
        #     posi_minus_radius = right_radius - softmax_logits_posi
        #     ood_score = posi_minus_radius.data.cpu().numpy()
        else:
            raise ValueError('Unknown open score type: {}'.format(self.cfg.TRAINER.NEGPROMPT.OPEN_SCORE))
        return predictions, ood_score, logits_posi, logits_negas

    # TODO: test_clip.test_nega_clip(), modify input and output
    def test(self, split=None): 
        '''
        From test_nega_clip in test_clip.py
        split: str, optional, "val" or "test"
        return results: dict with keys: ['ACC', 'AUPR', 'OSCR', 'FPR95', 'TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
        '''
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.test")
        # from SimpleTrainer.test()
        self.set_model_mode("eval")
        # self.evaluator.reset()    # TODO: no evaluator yet

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            testloader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            testloader = self.test_loader        
        # from test_nega_clip
        # correct, total = 0, 0
        _pred_k, _pred_u, _labels = [], [], []
        # logits_posi_id, logits_nega_id, logits_posi_ood, logits_nega_ood = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            # load trained text features (pos and neg?)
            if torch.cuda.device_count() > 1:
                prompts = self.model.module.prompt_learner()
                tokenized_prompts = self.model.module.tokenized_prompts
                text_features = self.model.module.text_encoder(prompts, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            else:
                prompts = self.model.prompt_learner()
                tokenized_prompts = self.model.tokenized_prompts
                text_features = self.model.text_encoder(prompts, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            torch.cuda.empty_cache()
            # breakpoint()

            # load test data: 1 for OOD, 0 for ID
            tqdm_object = tqdm(testloader, total=len(testloader))
            # batch test
            for self.batch_idx, batch in enumerate(tqdm_object):
                data, labels = self.parse_batch_test(batch)
                # get prediction logits
                if torch.cuda.device_count() > 1:
                    logits, _ = self.model.module.forward_test(data, text_features)
                    logits /= self.model.module.logit_scale.exp()
                else:
                    logits, _ = self.model.forward_test(data, text_features)
                    logits /= self.model.logit_scale.exp()
                # get metrics
                predictions, ood_score, logits_posi, logits_negas = self.get_ood_score(logits)
                # here the ood_score is in the shape of logits, it's hard to understand
                # _pred_k.append(ood_score)   # known class prediction

                # for all ood_score, if label is 0, add to pred_k, else add to pred_u
                # ood_score is size (batch_size, n_classes)
                # labels is size (batch_size,)
                # use pytorch's tensor operation to filter out the known and unknown classes
                _pred_k.append(ood_score[labels == 0].data.cpu().numpy())
                _pred_u.append(ood_score[labels == 1].data.cpu().numpy())



                # correct += (predictions == labels.data).sum()
                # total += labels.size(0)
                # _labels.append(labels.data.cpu().numpy())
                # logits_posi_id.append(logits_posi.data.cpu().numpy())
                # logits_nega_id.append(logits_negas.data.cpu().numpy())
            # acc = float(correct) * 100. / float(total)
            # print('Acc: {:.5f}'.format(acc))

            # # load test OOD data
            # # TODO: outloader is OOD data loader, it's not implemented in dataloader yet.
            # outloader = None
            # outloader = testloader  # this is temporary! for debugging
            # assert outloader is not None, "It seems that outloader is not implemented yet. Can't use test now."
            # tqdm_object = tqdm(outloader, total=len(outloader))
            # for self.batch_idx, batch in enumerate(tqdm_object):
            #     data, labels = self.parse_batch_test(batch)
                
            #     with torch.set_grad_enabled(False):
            #         if torch.cuda.device_count() > 1:
            #             logits, _ = self.model.module.forward_test(data, text_features)
            #             logits /= self.model.module.logit_scale.exp()
            #         else:
            #             logits, _ = self.model.forward_test(data, text_features)
            #             logits /= self.model.logit_scale.exp()
            #         predictions, ood_score, logits_posi, logits_negas = self.get_ood_score(logits)
            #         # here the ood_score is in the shape of logits, it's hard to understand
            #         _pred_u.append(ood_score)   # unknown class prediction
            #         # logits_posi_ood.append(logits_posi.data.cpu().numpy())
            #         # logits_nega_ood.append(logits_negas.data.cpu().numpy())

        # acc = float(correct) * 100. / float(total)
        # print('Acc: {:.5f}'.format(acc))

        _pred_k = np.concatenate(_pred_k, 0)
        _pred_u = np.concatenate(_pred_u, 0)
        # _labels = np.concatenate(_labels, 0)
        print('Shape of _pred_k: ', _pred_k.shape)  # (data size, n_classes)
        print('Shape of _pred_u: ', _pred_u.shape)  # (data size, n_classes)
        
        # Out-of-Distribution detction evaluation
        x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)   # get the max value of each row. shape: (data size,)
        # results = metric_ood(x1, x2)['Bas']
        # save _pred_k, -pred_u
        # score_dic = {}
        # score_dic['pred_k'] = _pred_k
        # score_dic['pred_u'] = _pred_u
        # score_dic['logits_posi_id'] = np.concatenate(logits_posi_id, 0)
        # score_dic['logits_nega_id'] = np.concatenate(logits_nega_id, 0)
        # score_dic['logits_posi_ood'] = np.concatenate(logits_posi_ood, 0)
        # score_dic['logits_nega_ood'] = np.concatenate(logits_nega_ood, 0)
        # np.save('savescores/' + options['dataset'] + '_ score_dic.npy', score_dic)
        # OSCR
        # _oscr_socre = compute_oscr(_pred_k, _pred_u, _labels)

        auroc, aupr, fpr95 = compute_fpr(x1, x2)
        # results['ACC'] = acc
        # results['OSCR'] = _oscr_socre * 100.
        # results['FPR95'] = fpr95 * 100.
        # results['AUPR'] = aupr * 100.   
        # print('ACC: {:.5f}, OSCR: {:.5f}, FPR95: {:.5f}, AUPR: {:.5f}, TNR: {:.5f}, AUROC: {:.5f}, DTACC: {:.5f}, AUIN: {:.5f}, AUOUT: {:.5f}'.format(
        #     results['ACC'], results['OSCR'], results['FPR95'], results['AUPR'], results['TNR'], results['AUROC'], results['DTACC'], results['AUIN'], results['AUOUT']))
        print('AUROC: {:.5f}, AUPR: {:.5f}, FPR95: {:.5f}'.format(auroc, aupr, fpr95))

        # draw tsne plot
        self.draw_tsne_plot(class_num=10)

        return auroc  # TODO: check what should be returned    

    def get_NND_loss(self, negative_text_features):
        '''
        Calculate the loss for negative-negative distance
        '''
        loss_nega_to_nega = 0
        for i in range(negative_text_features.shape[0]):    # for each class
            negative_features = negative_text_features[i,:,:].float()   # (n_nega_ctx , 512)
            negative_features_mean = torch.mean(negative_features, dim=0, keepdim=True)
            negative_features_mean_norm = negative_features_mean.norm(dim=-1, keepdim=True)  # (1, 1)

            # Euclidean distance
            # loss_nega_to_nega += -sum(torch.pdist(negative_features, p=2))

            # Cosine distance
            negative_features_norm = negative_features.norm(dim=-1, keepdim=True)   # (n_nega_ctx, 1)
            # nega_nega
            # dot_product = negative_features_norm @ negative_features_norm.t()
            # nega_mean
            dot_product = negative_features_norm @ negative_features_mean_norm.t()
            loss_nega_to_nega += -torch.mean(1-dot_product)
            loss_nega_to_nega /= negative_text_features.shape[0]
            return loss_nega_to_nega
    
    def get_NIS_loss(self, output, n_nega_ctx, labels):
        '''
        Calculate the loss for negative image similarity
        NOTE: by default use 'MSP' method, so comment out the 'Fence' and 'OE' method, so that no need to add cfg
        '''
        loss_nega_to_other = 0
        out_nega_forCE = output # [batch_size, nclass * 1+n_nega_ctx]
        # create soft_target(1-hot) for negative samples and positive samples
        soft_target = torch.zeros(out_nega_forCE.shape).long().cuda()
        idx = torch.arange(out_nega_forCE.shape[0]).cuda()
        # This means all classes are assigned an 1.
        soft_target.view(soft_target.shape[0], int(output.shape[1]/(1+n_nega_ctx)), -1)[idx, labels, :] = 1 # TODO: check what is this line doing
        # labels_nega = labels.reshape(1, -1).repeat(n_nega_ctx, 1).t().reshape(-1)
        # if options['open_set_method'] == 'MSP': # default
        loss_fun = nn.MultiLabelSoftMarginLoss(reduction='mean')
        loss_nega_to_other = loss_fun(out_nega_forCE, soft_target)
            # loss_nega_to_other = F.cross_entropy(out_nega_forCE, labels_nega)
        # elif options['open_set_method'] == 'Fence':
        #     loss_nega_to_other = custom_alpha_cross_entropy(out_nega_forCE, soft_target, alpha=options['fence_alpha'])
        # elif options['open_set_method'] == 'OE':    # may be reference
        #     loss_nega_to_other = -(out_nega_forCE.mean(1) - torch.logsumexp(out_nega_forCE, dim=1)).mean() #OE
        return loss_nega_to_other
    
    def get_NPD_loss(self, positive_text_features, negative_text_features):
        '''
        Calculate the loss for negative-positive distance
        NOTE: by default use 'MSP' method, so comment out the 'Fence' and 'OE' method, so that no need to add cfg
        NPD is more negative the better
        '''
        loss_nega_to_posi = 0
        all_class_dis = 0
        for i in range(negative_text_features.shape[0]):    # for each class
            positive_feature = positive_text_features[i:i+1,:].float()  # (1, 512)
            negative_feature = negative_text_features[i,:,:].float()    # (n_nega_ctx, 512)
            positive_feature_norm = positive_feature/positive_feature.norm(dim=-1, keepdim=True)
            negative_feature_norm = negative_feature/negative_feature.norm(dim=-1, keepdim=True)
            dot_product = positive_feature_norm @ negative_feature_norm.t()
            mean_cosine_dis = (1-dot_product).mean()
            all_class_dis += mean_cosine_dis
            
        # if options['open_set_method'] == 'MSP':
        loss_nega_to_posi -= all_class_dis/negative_text_features.shape[0]
        # elif options['open_set_method'] == 'Fence':
        #     loss_nega_to_posi = 0
        # else:
        #     loss_nega_to_posi += all_class_dis/negative_text_features.shape[0]
        return loss_nega_to_posi

    def forward_backward(self, batch):
        '''
        Run a batch, calculate loss (NND, NIS, NPD), update model and return loss
        Called in run_epoch
        ---
        batch: from iterating self.train_loader_x
        '''
        if DEBUG: 
            prev = self.model.prompt_learner.ctx_negative.data.clone()

        n_nega_ctx = self.cfg.TRAINER.NEGPROMPT.NEGA_CTX
        # get data and labels from batch (to device)
        image, labels = self.parse_batch_train(batch)   # TODO: make sure the data loader is correct   

        # NOTE: POMP is not used by default. Not move to here (so that not need to add config)
        with torch.set_grad_enabled(True):  # TODO: check here
            output, text_features = self.model(image)
            # output.shape = [batch_size, nclass * 1+n_nega_ctx]
            # text_features.shape = [nclass * (1+n_nega_ctx), 512]
            output_posi = output.view(-1, int(output.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)[:, :, 0]   # the first column is the logits for positive prompts for each class
            ensemble_text_features = text_features.view(int(text_features.shape[0]/(1+n_nega_ctx)), 1+n_nega_ctx, -1)   # shape = [n_class, 1+n_nega_ctx, 512]
            positive_text_features = ensemble_text_features[:, 0, :]    # shape = [n_class, 512]
            negative_text_features = ensemble_text_features[:, 1:, :]   # shape = [n_class, n_nega_ctx, 512]
            if DEBUG:
                print(f"Output shape is {output.shape}, should be [batch_size, {self.n_cls*(1+n_nega_ctx)}]")
                print(f"Text features shape is {text_features.shape}, should be [{self.n_cls*(1+n_nega_ctx)}, 512]")
                print(f"Output positive shape is {output_posi.shape}, should be [{self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE}, {self.n_cls}]")
                print(f"Positive text features shape is {positive_text_features.shape}, should be [{self.n_cls}, 512]")
                print(f"Negative text features shape is {negative_text_features.shape}, should be [{self.n_cls}, {n_nega_ctx}, 512]")

            # calculate loss
            # 0. CE loss for pos prompt (actually no used here, just to keep original code)
            loss_positive = F.cross_entropy(output_posi, labels)
            # loss_positive *= 1e-8   # not important
            # NOTE: prototype loss is not used by default. Not move to here.

            # 1. NND loss: negative-negative distance
            # loss_nega_to_nega = self.get_NND_loss(negative_text_features)
            loss_nega_to_nega = torch.tensor(0.0).cuda()
            for i in range(negative_text_features.shape[0]):    # for each class
                negative_features = negative_text_features[i,:,:].float()   # (n_nega_ctx , 512)
                negative_features_mean = torch.mean(negative_features, dim=0, keepdim=True)
                negative_features_mean_norm = negative_features_mean.norm(dim=-1, keepdim=True)  # (1, 1)

                # Euclidean distance
                # loss_nega_to_nega += -sum(torch.pdist(negative_features, p=2))

                # Cosine distance
                negative_features_norm = negative_features.norm(dim=-1, keepdim=True)   # (n_nega_ctx, 1)
                # nega_nega
                # dot_product = negative_features_norm @ negative_features_norm.t()
                # nega_mean
                dot_product = negative_features_norm @ negative_features_mean_norm.t()
                loss_nega_to_nega += -torch.mean(1-dot_product)
            loss_nega_to_nega /= negative_text_features.shape[0]

            if DEBUG: print(f"Loss NND done")

            # 2. NIS loss: Negative image similarity
            # NOTE: by default use 'MSP' method, so comment out the 'Fence' and 'OE' method, so that no need to add cfg
            # loss_nega_to_other = self.get_NIS_loss(output, n_nega_ctx, labels)
            loss_nega_to_other = torch.tensor(0.0).cuda()
            out_nega_forCE = output # [batch_size, nclass * 1+n_nega_ctx]
            # create soft_target(1-hot) for negative samples and positive samples
            soft_target = torch.zeros(out_nega_forCE.shape).long().cuda()
            idx = torch.arange(out_nega_forCE.shape[0]).cuda()
            # This means all classes are assigned an 1.
            soft_target.view(soft_target.shape[0], int(output.shape[1]/(1+n_nega_ctx)), -1)[idx, labels, :] = 1 # TODO: check what is this line doing
            # labels_nega = labels.reshape(1, -1).repeat(n_nega_ctx, 1).t().reshape(-1)
            # if options['open_set_method'] == 'MSP': # default
            loss_fun = nn.MultiLabelSoftMarginLoss(reduction='mean')
            loss_nega_to_other = loss_fun(out_nega_forCE, soft_target)
            if DEBUG: print(f"Loss NIS done")

            # 3. NPD loss: Negative-Positive Distance
            # NOTE: by default use 'MSP' method, so comment out the 'Fence' and 'OE' method, so that no need to add cfg
            # loss_nega_to_posi = self.get_NPD_loss(positive_text_features, negative_text_features)
            loss_nega_to_posi = torch.tensor(0.0).cuda()
            all_class_dis = torch.tensor(0.0).cuda()
            for i in range(negative_text_features.shape[0]):    # for each class
                positive_feature = positive_text_features[i:i+1,:].float()  # (1, 512)
                negative_feature = negative_text_features[i,:,:].float()    # (n_nega_ctx, 512)
                positive_feature_norm = positive_feature/positive_feature.norm(dim=-1, keepdim=True)
                negative_feature_norm = negative_feature/negative_feature.norm(dim=-1, keepdim=True)
                dot_product = positive_feature_norm @ negative_feature_norm.t()
                mean_cosine_dis = (1-dot_product).mean()
                all_class_dis += mean_cosine_dis
                
            # if options['open_set_method'] == 'MSP':
            loss_nega_to_posi -= all_class_dis/negative_text_features.shape[0]

            if DEBUG: print(f"Loss NPD done")
            if DEBUG: print(f'loss weight: NIS {self.cfg.TRAINER.NEGPROMPT.NETATIVE_WEIGHT}, NND {self.cfg.TRAINER.NEGPROMPT.NEGA_NEGA_WEIGHT}, NPD {self.cfg.TRAINER.NEGPROMPT.DISTANCE_WEIGHT}')

            # aggregate loss: weighted by cfg. prototype loss not used here
            loss = loss_nega_to_other*self.cfg.TRAINER.NEGPROMPT.NETATIVE_WEIGHT \
                + loss_nega_to_nega*self.cfg.TRAINER.NEGPROMPT.NEGA_NEGA_WEIGHT \
                + loss_nega_to_posi*self.cfg.TRAINER.NEGPROMPT.DISTANCE_WEIGHT 
            
            # TODO: NOTE: temp: for debug purpose!!!
            # output_nega = output.view(-1, int(output.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)[:, :, 1]
            # loss = F.cross_entropy(output_nega, labels)
            # loss = text_features.mean()

            # clone original prompt_learner.ctx_negative
            neg_prompt_before_update = deepcopy(self.model.prompt_learner.ctx_negative).detach()

            # backward and update
            self.model_backward_and_update(loss)
           
           
            if DEBUG: 
                print(f"IF CTX_NEG CHANGED: {not torch.equal(prev, self.model.prompt_learner.ctx_negative.data)}")
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad:  # 仅检查 requires_grad=True 的参数
                        if param.grad is None:
                            print(f"{name} has no gradient!")
                        else:
                            print(f"{name} gradient mean: {param.grad.mean()}")


                # Check if ctx_negative has changed
                if torch.equal(self.model.prompt_learner.ctx_negative, neg_prompt_before_update):
                    print("neg prompt has not changed.")
                else:
                    print("neg prompt has changed!!")

        loss_summary = {
            "loss": loss.item(),
            "loss_positive": loss_positive.item(),
            "NIS Loss": loss_nega_to_other.item(),
            "NND Loss": loss_nega_to_nega.item(),
            "NPD Loss": loss_nega_to_posi.item(),
            # "acc": compute_accuracy(output, labels)[0].item(),    # no acc here
        }
        if DEBUG: print(f"Loss summary: {loss_summary}")

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    # modify coop's function for negprompt
    def parse_batch_train(self, batch):
        '''Parse batch for training, also move to device
        Called in forward_backward
        '''
        ''' A typical data loader in NegPrompt: 
        single item: (sample, target) where target is class_index of the target class.
        batch input:
        data shape = [batch_size, 3, 224, 224]
        labels shape = [batch_size,]
        '''
        input = batch["img"]
        if DEBUG: print(f"input shape is {input.shape}, should be [{self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE}, 3, 224, 224]")
        label = batch["label"]
        if DEBUG: print(f"labels shape is {label.shape}, should be [{self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE},]")
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def draw_tsne_plot(self, epoch=None, class_num=10, labels_to_draw=None):
        '''
        Draw tsne plot for pos and neg text features and image features
        class_num: number of classes to draw, if not specify class_to_draw
        labels_to_draw: list of int, specify the class index to draw, if not None, class_num will be ignored
        '''
        # think of using training data to draw tsne plot, draw OOD samples from test OOD data
        prompts = self.model.prompt_learner()   # call forward_negative, return both positive and negative prompts, shape = [nclass, 1+n_nega, n_ctx, 512]
        tokenized_prompts = self.model.tokenized_prompts # shape = [nclass* (1+n_nega), n_ctx]
        text_features = self.model.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.reshape(prompts.shape[0], prompts.shape[1], text_features.shape[-1])  # shape = [nclass, 1+n_nega, 512]
        pos_feature = text_features[:, 0:1, :].cpu()
        pos_feature = pos_feature / pos_feature.norm(dim=-1, keepdim=True)
        neg_feature = text_features[:, 1:, :].cpu()
        neg_feature = neg_feature / neg_feature.norm(dim=-1, keepdim=True)
        pos_label = torch.arange(pos_feature.shape[0])[..., None] # shape = [nclass, 1]
        neg_label = torch.full((neg_feature.shape[0], neg_feature.shape[1]), pos_feature.shape[0]) #shape = [nclass, n_nega], value = nclass
        print('get text features done')

        # get labels to draw
        n_class = pos_feature.shape[0]
        if labels_to_draw is not None:
            assert all([label < n_class for label in labels_to_draw]), "label_to_draw should be within n_class"
            labels_to_draw = torch.Tensor(labels_to_draw).to(self.device)   # cuda
            class_num = len(labels_to_draw)
        else:
            # sample class_num classes
            class_num = min(class_num, n_class)
            labels_to_draw = torch.randperm(n_class)[:class_num]
            labels_to_draw = labels_to_draw.to(self.device)
        print('numbers of classes to draw: ', class_num)
        print(f'labels to draw: {labels_to_draw}')
        
        # get all image features of ID samples
        all_image_feature = torch.Tensor()
        all_image_label = torch.Tensor()
        print('getting ID image features...')
        tqdm_object = tqdm(self.train_loader_x, total=len(self.train_loader_x))
        for batch in tqdm_object:
            data, labels = self.parse_batch_train(batch)
            # filter labels to draw
            mask = torch.isin(labels, labels_to_draw)
            data = data[mask]
            labels = labels[mask]
            with torch.set_grad_enabled(False):
                image_features = self.model.image_encoder(data.type(self.model.dtype)).cpu()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_image_feature = torch.cat([all_image_feature, image_features], dim=0)
                all_image_label = torch.cat([all_image_label, labels.cpu()], dim=0)

        # get number of ID samples to draw: only plot OOD samples no more than ID samples
        n_id_samples = all_image_feature.shape[0]

        # get all image features of OOD samples
        print('getting OOD image features...')
        tqdm_object = tqdm(self.test_loader, total=len(self.test_loader))
        n_ood_samples = 0
        for batch in tqdm_object:
            data, labels = self.parse_batch_train(batch)
            # keep only label == 1
            data = data[labels == 1]
            # make all labels nclass
            labels = torch.full((data.shape[0],), n_class)
            n_ood_samples += data.shape[0]
            with torch.set_grad_enabled(False):
                image_features = self.model.image_encoder(data.type(self.model.dtype)).cpu()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_image_feature = torch.cat([all_image_feature, image_features], dim=0)
                all_image_label = torch.cat([all_image_label, labels.cpu()], dim=0)
            if n_ood_samples > n_id_samples:
                break
        
        # get filtered text features and filtered labels
        labels_to_draw = labels_to_draw.cpu()
        pos_feature = pos_feature[labels_to_draw]
        neg_feature = neg_feature[labels_to_draw]
        pos_label = pos_label[labels_to_draw]
        neg_label = neg_label[labels_to_draw]

        all_text_feature = torch.Tensor()               
        all_text_feature = torch.cat([all_text_feature, pos_feature, neg_feature], dim=1)
        all_text_feature = all_text_feature.view(-1, all_text_feature.shape[-1])
        
        all_text_label = torch.Tensor()
        all_text_label = torch.cat([all_text_label, pos_label, neg_label], dim=1)
        all_text_label = all_text_label.view(-1)
        
        total_feature = torch.cat([all_text_feature, all_image_feature], dim=0)
        total_label = torch.cat([all_text_label, -1 * (all_image_label+1)], dim=0)

        print('getting tsne features...')
        X = total_feature.detach().numpy()
        tsne_model = TSNE(metric="precomputed", n_components=2, init="random", perplexity=30)
        distance_matrix = pairwise_distances(X, X, metric='cosine', n_jobs=-1)
        
        data = torch.Tensor(tsne_model.fit_transform(distance_matrix))
        target = total_label
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=256)
        plt.figure()
        print('drawing tsne plot...')
        for x, y in loader:
            # 样本点显示
            idx_pos_text = (y < n_class) & (y >= 0)  # 正向样本 
            idx_nega_text = (y >= n_class)  # 负向样本
            idx_pos_image = (y < 0) & (y >= -n_class)
            idx_nega_image = (y < -n_class)

            plt.axis('off')  # Remove axes
            plt.scatter(x[idx_pos_text, 0], x[idx_pos_text, 1], marker = 'o', c=y[idx_pos_text], alpha=0.7,
                        cmap=plt.cm.get_cmap("plasma", n_class + 1), label='Positive Text')
            plt.scatter(x[idx_nega_text, 0], x[idx_nega_text, 1], marker = 'o', c=y[idx_nega_text], alpha=0.7,
                        cmap=plt.cm.get_cmap("summer", n_class + 1), label='Negative Text')
            plt.scatter(x[idx_pos_image, 0], x[idx_pos_image, 1], marker = 'x',c =-1 * y[idx_pos_image] - 1, alpha=0.5,
                        cmap=plt.cm.get_cmap("plasma", n_class + 1), label='ID Images')
            plt.scatter(x[idx_nega_image, 0], x[idx_nega_image, 1], marker = 'x',c=-1 * y[idx_nega_image] - 1, alpha=0.7,
                        cmap=plt.cm.get_cmap("summer", n_class + 1), label='OOD Images')
        
        # Add legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()  # Adjust layout to prevent legend cutoff
        plt.savefig(os.path.join(self.output_dir, 'tsne_plot.png'), bbox_inches='tight')
        plt.close()
        print(f'tsne plot saved to {os.path.join(self.output_dir, "tsne_plot.pdf")}')


    # Qi: 
    # TODO: need to check if we really need to override this method
    # NOTE: this is from trainerX template

    # def run_epoch(self):
    #     self.set_model_mode("train")
    #     losses = MetricMeter()
    #     batch_time = AverageMeter()
    #     data_time = AverageMeter()
    #     self.num_batches = len(self.train_loader_x)

    #     end = time.time()
        
    #     for self.batch_idx, batch in enumerate(self.train_loader_x):
    #         data_time.update(time.time() - end)
    #         # forward, update, and get loss
    #         loss_summary = self.forward_backward(batch)
    #         batch_time.update(time.time() - end)
    #         losses.update(loss_summary)

    #         meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
    #         only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
    #         if meet_freq or only_few_batches:
    #             nb_remain = 0
    #             nb_remain += self.num_batches - self.batch_idx - 1
    #             nb_remain += (
    #                 self.max_epoch - self.epoch - 1
    #             ) * self.num_batches
    #             eta_seconds = batch_time.avg * nb_remain
    #             eta = str(datetime.timedelta(seconds=int(eta_seconds)))

    #             info = []
    #             info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
    #             info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
    #             info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
    #             info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
    #             info += [f"{losses}"]
    #             info += [f"lr {self.get_current_lr():.4e}"]
    #             info += [f"eta {eta}"]
    #             print(" ".join(info))

    #         n_iter = self.epoch * self.num_batches + self.batch_idx
    #         for name, meter in losses.meters.items():
    #             self.write_scalar("train/" + name, meter.avg, n_iter)
    #         self.write_scalar("train/lr", self.get_current_lr(), n_iter)

    #         end = time.time()
    
    # NOTE: template from Dassl, may not need to modify
    # def after_epoch(self):
    #     pass    # TODO: remove this when test is implemented
    #     last_epoch = (self.epoch + 1) == self.max_epoch
    #     do_test = not self.cfg.TEST.NO_TEST
    #     meet_checkpoint_freq = (
    #         (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
    #         if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
    #     )

    #     if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
    #         curr_result = self.test(split="val")
    #         is_best = curr_result > self.best_result
    #         if is_best:
    #             self.best_result = curr_result
    #             self.save_model(
    #                 self.epoch,
    #                 self.output_dir,
    #                 val_result=curr_result,
    #                 model_name="model-best.pth.tar"
    #             )

    #     if meet_checkpoint_freq or last_epoch:
    #         self.save_model(self.epoch, self.output_dir)

    # template from Dassl
    # TODO: add drawing tsne graph
    # def after_train(self):
    #     pass
        #print("Finish training")

        # do_test = not self.cfg.TEST.NO_TEST
        # if do_test:
        #     if self.cfg.TEST.FINAL_MODEL == "best_val":
        #         print("Deploy the model with the best val performance")
        #         self.load_model(self.output_dir)
        #     else:
        #         print("Deploy the last-epoch model")
        #     self.test()

        # # Show elapsed time
        # elapsed = round(time.time() - self.time_start)
        # elapsed = str(datetime.timedelta(seconds=elapsed))
        # print(f"Elapsed: {elapsed}")

        # # Close writer
        # self.close_writer()

