 main model code for negprompt

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

# A global boolean for local debugging usage
# should be False for local debugging on CPU, True for training 
CUDA_ENABLED = torch.cuda.is_available()

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

# Ensure the device is set globally, 解决CPU、CUDA使用问题 Yuhao add
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # TODO: further check if forward_positive is needed. If not discard it
    def forward_positive(self):
        pass
    
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

    def update_ctx_positive(self, ctx_posi):
        pass
    
    def update_ctx_negative(self, ctx_nega):
        pass

    def freeze_ctx_positive(self):
        pass

    def get_ctx_positive(self):
        pass

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
    
    # override SimpleTrainer的train()
    # Yuhao: framework copied from train in TrainerBase, need to keep work on run_epoch(), after_epoch(), after_train(), ...
    def train(self.start_epoch, self.max_epoch): 
        print("Calling CoOp_works\\CoOp\\trainers\\negprompt.NegPrompt.train")
        self.before_train() # 在SimpleTrainer中，感觉无结构性作用可照搬
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.run_epoch(self.epoch, self.max_epoch)
            self.after_epoch()
        self.after_train()

        # TODO


    # Essential! 从NegPrompt main_worker(), for epoch 下copy
    def run_epoch(self, epoch, max_epoch):
        last_loss = 9999999999
        print("==> Epoch {}/{}".format(epoch+1, max_epoch))
        # TODO
        # self.model, self.optim, self.sched built in build_model(), need to keep work _
        this_loss = self.train_nega_clip(self.model, self.optim, self.sched, trainloader, run, epoch=epoch, proto = proto, **options)
        this_loss = round(this_loss, 8)
        print('this : ', this_loss)
        if this_loss == last_loss:
            print('the same')
            break
        last_loss = this_loss
        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results = test_nega_clip(model, criterion, testloader, outloader, epoch=0, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t FPR95 (%): {:.3f}\t AUPR (%): {:.3f}\t".format(
                 results['ACC'], results['AUROC'], results['OSCR'], results['FPR95'], results['AUPR']))
            # results = test_clip(model, criterion, testloader, outloader, epoch=0, **options)
            # print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
            run.log(results, step = epoch)
            if results['ACC'] > best_acc and options['LOG'] and options['stage'] == 1:
                best_acc = results['ACC']
                save_networks(model, model_path, file_name)
            if results['AUROC'] > best_auroc and options['LOG'] and options['stage'] == 3:
                best_auroc = results['AUROC']
                print("save:", model_path)
                save_networks(model, model_path, file_name)
            if results['AUROC'] > best_auroc:
                best_auroc = results['AUROC']
            run.log({'best_auroc': best_auroc}, step = epoch)
        if options['stepsize'] > 0: scheduler.step()
        # draw the t-sne plot of all the text features
        if 'ImageNet' not in options['dataset']:
            model.draw_tsne_plot(testloader, outloader, options['outf'], expr_name, epoch)  
        print('Now running stage_{}, dataset_{}, best_auroc: {}'.format(options['stage'], options['dataset'], best_auroc))
        if options['stage'] == 4:
            print('Original dataset : ', options['ori_dataset'])
    
    # TODO
    def after_epoch(self):
      pass


    # TODO
    # Yuhao：copied from NegPrompt/core/train_clip.py.
    # usage:  train_nega_clip(model, optimizer, scheduler, trainloader, run, epoch=epoch, proto = proto, **options)
    def train_nega_clip(net, optimizer, scheduler, trainloader, run, epoch=None,  proto=None, **options):
        '''Trains the given network using the provided optimizer, scheduler, and data loader. For 1 epoch.
        This function supports various training options including the use of negative context, prototype weighting, and different open set methods.
        Args:
            net (torch.nn.Module): The model to be trained. NegaPromptCLIP
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
            trainloader (torch.utils.data.DataLoader): DataLoader for the training data.
            run (object): Object for logging the training process.
            epoch (int, optional): The current epoch number. Defaults to None.
            proto (torch.Tensor, optional): Prototype tensor for prototype loss calculation. Defaults to None. This is the average Image Embedding for each class
            **options: Additional options for training, including:
                - 'NEGA_CTX' (int): Number of negative contexts.
                - 'use_gpu' (bool): Flag to use GPU if available.
                - 'POMP' (bool): Flag to use POMP method. May stands for 'xxx -Oriented Multi-Prototype' method.
                - 'POMP_k' (int): Parameter for POMP method.
                - 'num_classes' (int): Number of classes in the dataset.
                - 'prototype_weight' (float): Weight for the prototype loss.
                - 'stage' (int): Training stage.
                - 'open_set_method' (str): Method for open set recognition ('MSP', 'Fence', 'OE').
                - 'fence_alpha' (float): Alpha parameter for Fence method.
                - 'negative_weight' (float): Weight for the negative loss.  (NID)?  # TODO to confirm
                - 'distance_weight' (float): Weight for the distance loss.  (NPD)
                - 'nega_nega_weight' (float): Weight for the negative-to-negative loss. (NND)
                - 'print_freq' (int): Frequency of printing the training status.
        Returns:
            float: The average loss over the training data.
        '''
        losses = AverageMeter()
        loss_all = 0
        n_nega_ctx = options['NEGA_CTX']

        # train for one epoch, 
        for batch_idx, (data, labels) in enumerate(trainloader):
            if options['use_gpu']:
              data, labels = data.cuda(), labels.cuda()
        
            # TODO: figure out what is POMP and how it is used
            if options['POMP']: # ori = orignial tag, modify_to_ori is a dic that transform the modified labels to original ones
                # it's a k-number to k-number mapping
                ori_to_modify, modify_to_ori = label_transform(labels.cpu().numpy(), options['POMP_k'], options['num_classes']-1)
                modified_labels = torch.tensor([ori_to_modify[label.item()] for label in labels]).cuda()
                labels = modified_labels    # from 0 to k-1
            else:
                ori_to_modify, modify_to_ori = None, None
            
            # calculate the loss and update the model
            with torch.set_grad_enabled(True):
                # get the logits and text embeddings
                output, text_features = net(data, modify_to_ori)    # logits (represents similarity) and text embeddings by CLIP text encoder
                # output.shape = [batch_size, nclass * 1+n_nega_ctx]
                # text_features.shape = [nclass * (1+n_nega_ctx), 512]
                output_posi = output.view(-1, int(output.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)[:, :, 0]   # the first column is the logits for positive prompts for each class
                ensemble_text_features = text_features.view(int(text_features.shape[0]/(1+n_nega_ctx)), 1+n_nega_ctx, -1)   # shape = [n_class, 1+n_nega_ctx, 512]
                positive_text_features = ensemble_text_features[:, 0, :]    # shape = [n_class, 512]
                negative_text_features = ensemble_text_features[:, 1:, :]   # shape = [n_class, n_nega_ctx, 512]
        
                # the classification loss
                loss_positive = F.cross_entropy(output_posi, labels)
                loss_prototype = 0
                if(options['prototype_weight'] != 0):
                    loss_prototype = -torch.sum(torch.mul(positive_text_features, proto))   # this evaluates the cosine similarity between the positive text features and the prototype
                    
                # calculate 
                loss_nega_to_other = 0  # this maybe the NID loss
                loss_nega_to_posi = 0
                loss_nega_to_nega = 0
                
                if options['stage'] > 1:
                    loss_positive *= 1e-8   # not important
                    # negative_features = negative_features.view(0)
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
                    
                    # calculate the NID loss
                    # print(output_negas.transpose(1,2))
                    out_nega_forCE = output # [batch_size, nclass * 1+n_nega_ctx]
                    # create soft_target(1-hot) for negative samples and positive samples
                    soft_target = torch.zeros(out_nega_forCE.shape).long().cuda()
                    idx = torch.arange(out_nega_forCE.shape[0]).cuda()
                    # This means all classes are assigned an 1.
                    soft_target.view(soft_target.shape[0], int(output.shape[1]/(1+n_nega_ctx)), -1)[idx, labels, :] = 1 # TODO: check what is this line doing
                    # labels_nega = labels.reshape(1, -1).repeat(n_nega_ctx, 1).t().reshape(-1)
                    if options['open_set_method'] == 'MSP':
                        loss_fun = nn.MultiLabelSoftMarginLoss(reduction='mean')
                        loss_nega_to_other = loss_fun(out_nega_forCE, soft_target)
                        # loss_nega_to_other = F.cross_entropy(out_nega_forCE, labels_nega)
                    elif options['open_set_method'] == 'Fence':
                        loss_nega_to_other = custom_alpha_cross_entropy(out_nega_forCE, soft_target, alpha=options['fence_alpha'])
                    elif options['open_set_method'] == 'OE':
                        loss_nega_to_other = -(out_nega_forCE.mean(1) - torch.logsumexp(out_nega_forCE, dim=1)).mean() #OE
                    # elif options['open_set_method'] == 'Wasserstein':
                    #     labels_openset = torch.eye(output_negas.shape[1], output_negas.shape[1]).cuda()
                    #     labels_openset = labels_openset.unsqueeze(-1)
                    #     #softmax out_nega_forCE
                    #     sm_out = F.softmax(out_nega_forCE, dim=1).cuda()
                    #     wass_nega = sm_out.unsqueeze(-1)
                    #     wood_loss = SamplesLoss(loss="sinkhorn", diameter=1., p=2, blur=1., cost = custom_cost)
                    #     batch_size = wass_nega.shape[0]
                    #     wass_loss = torch.zeros(batch_size, output_negas.shape[1]).cuda()
                    #     for b in range(batch_size):
                    #         input_b = wass_nega[b:b+1, :, :].repeat(output_negas.shape[1], 1, 1).float().cuda()
                    #         wass_loss[b] = wood_loss(input_b[:,:,0], input_b, labels_openset[:,:,0], labels_openset)
                    #     values, idx = torch.min(wass_loss, dim=1)
                    #     loss_nega_to_other = -torch.mean(values)
                    else:
                        raise NotImplementedError
                    
                    # calculate the NPD loss, similar to the NND loss
                    all_class_dis = 0
                    for i in range(negative_text_features.shape[0]):    # for each class
                        positive_feature = positive_text_features[i:i+1,:].float()  # (1, 512)
                        negative_feature = negative_text_features[i,:,:].float()    # (n_nega_ctx, 512)
                        positive_feature_norm = positive_feature/positive_feature.norm(dim=-1, keepdim=True)
                        negative_feature_norm = negative_feature/negative_feature.norm(dim=-1, keepdim=True)
                        dot_product = positive_feature_norm @ negative_feature_norm.t()
                        mean_cosine_dis = (1-dot_product).mean()
                        all_class_dis += mean_cosine_dis
                        
                    if options['open_set_method'] == 'MSP':
                        loss_nega_to_posi -= all_class_dis/negative_text_features.shape[0]
                    elif options['open_set_method'] == 'Fence':
                        loss_nega_to_posi = 0
                    else:
                        loss_nega_to_posi += all_class_dis/negative_text_features.shape[0]
                    
                loss = loss_positive + options['prototype_weight'] * loss_prototype \
                        + options['negative_weight']*loss_nega_to_other + options['distance_weight']*loss_nega_to_posi + options['nega_nega_weight']*loss_nega_to_nega

                net.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            losses.update(loss.item(), labels.size(0))
        
            if (batch_idx+1) % options['print_freq'] == 0: 
                print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                      .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
            
            loss_all += losses.avg
        run.log({'loss': loss_all}, step=epoch)
        return loss_all