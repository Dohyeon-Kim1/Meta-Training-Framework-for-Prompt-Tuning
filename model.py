import torch
import torch.nn as nn
import torch.nn.functional as F

from clip.clip import _MODELS, _download, build_model, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()


def load_clip_to_cpu():
    backbone_name = "ViT-B/16"
    url = _MODELS[backbone_name]
    model_path = _download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        # self.prompt_fc = nn.Linear(ctx_dim, ctx_dim, bias=False, dtype=clip_model.dtype)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if False:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"
        self.class_names = classnames

    def forward(self):
        ctx = self.ctx
        # ctx = self.prompt_fc(ctx)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ProtoAttnCoOp(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        # self.clip = clip_model
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = nn.Parameter(clip_model.logit_scale)
        self.cross_attn = nn.MultiheadAttention(embed_dim=clip_model.ln_final.weight.shape[0], 
                                                num_heads=8, batch_first=True, dtype=clip_model.dtype)
        self.meta_mlp = nn.Sequential(nn.Linear(clip_model.ln_final.weight.shape[0], clip_model.ln_final.weight.shape[0]*4, dtype=clip_model.dtype),
                                      nn.ReLU(),
                                      nn.Linear(clip_model.ln_final.weight.shape[0]*4, clip_model.ln_final.weight.shape[0], dtype=clip_model.dtype))
        self.dtype = clip_model.dtype
        self.args = args

        for name, param in self.named_parameters():
            if "prompt_learner" not in name and "cross_attn" not in name and "logit_scale" not in name and "meta_mlp" not in name:
                param.requires_grad_(False)

    def forward(self, spt_image, qry_image, label):
        N, K, C, H, W = spt_image.shape
        spt_image_features = self.image_encoder(spt_image.type(self.dtype).view(-1,C,H,W))
        qry_image_features = self.image_encoder(qry_image.type(self.dtype).view(-1,C,H,W))

        # hard_prompt = torch.cat([tokenize(f"a photo of a {c}") for c in label]).to(spt_image_features.device)
        # hard_text_features = self.clip.encode_text(hard_prompt)

        idx = [self.prompt_learner.class_names.index(n) for n in label]
        prompts = self.prompt_learner()[idx]
        tokenized_prompts = self.tokenized_prompts[idx]
        text_features = self.text_encoder(prompts, tokenized_prompts)

        spt_image_features = spt_image_features.view(N,K,-1)
        text_features = text_features.unsqueeze(1)

        spt_image_features = spt_image_features / spt_image_features.norm(dim=-1, keepdim=True)
        qry_image_features = qry_image_features / qry_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # adjusted_features = self.meta_mlp(spt_image_features.mean(dim=1)) + text_features.squeeze(1)
        # adjusted_features = adjusted_features / adjusted_features.norm(dim=-1, keepdim=True)


        adjusted_features = self.cross_attn(text_features, spt_image_features, spt_image_features)[0] + text_features
        # adjusted_features = adjusted_features.squeeze(1)
        adjusted_features = (adjusted_features / adjusted_features.norm(dim=-1, keepdim=True)).squeeze(1)
        adjusted_features = self.meta_mlp(adjusted_features) + adjusted_features
        adjusted_features = adjusted_features / adjusted_features.norm(dim=-1, keepdim=True)

        # soft_text_features = soft_text_features / soft_text_features.norm(dim=-1, keepdim=True)

        return adjusted_features, qry_image_features, text_features.squeeze(1)
    
    def get_image_feat(self, img):
        img_feat = self.image_encoder(img.type(self.dtype))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        return img_feat
    
    def get_text_feat(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_feat = self.text_encoder(prompts, tokenized_prompts)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return text_feat

    
    def get_meta_loss(self, adjusted_features, image_features, text_features=None):
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ adjusted_features.t()

        label = torch.LongTensor([i for i in range(self.args.n_way) for j in range(self.args.n_qry)]).to(device=logits.device)
        loss = F.cross_entropy(logits, label)
        if text_features is not None:
            target_logits = logit_scale * image_features @ text_features.t()
            loss += 0.2 * F.cross_entropy(target_logits, F.softmax(logits, dim=1))

        return loss
    
    def get_meta_acc(self, text_features, image_features):
        pred = (image_features @ text_features.t()).argmax(dim=-1)
        label = torch.LongTensor([i for i in range(self.args.n_way) for j in range(self.args.n_qry)]).to(device=pred.device)
        acc = (pred == label).sum() / len(label)

        return acc
    
    def get_acc(self, image_features, text_features, label):
        pred = (image_features @ text_features.t()).argmax(dim=-1)
        acc = (pred == label).sum() / len(label)
        
        return acc


class CoOp(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        for name, param in self.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features

    def get_image_feat(self, img):
        img_feat = self.image_encoder(img.type(self.dtype))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        return img_feat
    
    def get_text_feat(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_feat = self.text_encoder(prompts, tokenized_prompts)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return text_feat

    def get_loss(self, image_features, text_features, label):
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        loss = F.cross_entropy(logits, label)
        return loss
    
    def get_acc(self, image_features, text_features, label):
        pred = (image_features @ text_features.t()).argmax(dim=-1)
        acc = (pred == label).sum() / len(label)
        return acc