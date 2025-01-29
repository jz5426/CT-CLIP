import math
import copy
from contextlib import contextmanager
from functools import partial, wraps
from pathlib import Path

from cxr_clip_utils import load_cxr_clip_image_encoder
from gloria_utils import GloRIaVisionModel, GloRIaVisionModelDenseNet, GloRIaVisionModelResNet
from medclip_utils import MedCLIPVisionModel, MedCLIPVisionModelResNet, MedCLIPVisionModelViT
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from ct_clip.mlm import MLM
from ct_clip.visual_ssl import SimSiam, SimCLR

import warnings
# helper functions

def identity(t, *args, **kwargs):
    return t

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# checkpointing helper function

def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# helper classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w z) c -> b c h w z', h = h_r, w= w_r)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# patch dropout

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x, force_keep_all = False):
        if not self.training or self.prob == 0. or force_keep_all:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

# rotary positional embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device = device).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

# transformer

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias = False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, causal = False, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias = False), LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None, rotary_pos_emb = None):
        h, device, scale = self.heads, x.device, self.scale

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        if exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            dim_head = 64,
            heads = 8,
            causal = False,
            attn_dropout = 0.,
            ff_dropout = 0.,
            ff_mult = 4,
            checkpoint_during_training = False
    ):
        super().__init__()
        self.checkpoint_during_training = checkpoint_during_training

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult)),
            ]))

        self.norm_in = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)

    def forward(
            self,
            x,
            rotary_pos_emb = None,
            mask = None
    ):
        can_checkpoint = self.training and self.checkpoint_during_training
        checkpoint_fn = make_checkpointable if can_checkpoint else identity

        x = self.norm_in(x)

        for attn, ff in self.layers:
            attn, ff = map(checkpoint_fn, (attn, ff))

            x = attn(x, mask, rotary_pos_emb) + x
            x = ff(x) + x

        return self.norm_out(x)

# text and vision transformers

class TextTransformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            num_tokens,
            max_seq_len,
            dim_head,
            rotary_pos_emb = None,
            causal = False,
            **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if not rotary_pos_emb else None
        self.rotary_pos_emb = RotaryEmbedding(min(dim_head, 32)) if rotary_pos_emb else None

        self.cls_token = nn.Parameter(torch.randn(dim)) if not causal else None

        self.transformer = Transformer(dim, dim_head = dim_head, causal = causal, **kwargs)

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            rotary_pos_emb = self.rotary_pos_emb(n + 1, device = device)

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim = 1)

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

        out = self.transformer(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
        return out

class VisionTransformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            image_size,
            patch_size,
            channels,
            patch_dropout = 0.5,
            **kwargs
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        self.pos_emb = nn.Embedding(num_patches, dim)
        self.patch_dropout = PatchDropout(patch_dropout)

        self.transformer = Transformer(dim, **kwargs)

        self.to_cls_tokens = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, dim, bias = False),
            Rearrange('b d -> b 1 d')
        )

    def forward(
            self,
            x,
            keep_all_patches = False
    ):
        device = x.device

        x = self.to_tokens(x)
        b, n, _ = x.shape

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        x = self.patch_dropout(x, force_keep_all = keep_all_patches)

        out = self.transformer(x)

        cls_tokens = self.to_cls_tokens(out)
        return torch.cat((cls_tokens, out), dim = 1)

# contrastive learning functions

def model_forward_with_context(
        *,
        fn,
        args,
        freeze,
):
    encoding_context = null_context if not freeze else torch.no_grad

    with encoding_context():
        enc = fn(*args)

        if freeze:
            enc.detach_()

    return enc

# main clip class

class CTCLIP(nn.Module):
    def __init__(
            self,
            *,
            image_encoder = None,
            text_encoder = None,
            # tokenizer = None,
            dim_text = 512,
            dim_image = 512,
            dim_latent = 512,
            num_text_tokens = 28897,
            text_enc_depth = 6,
            text_seq_len = 256,
            text_heads = 8,
            text_dim_head = 64,
            text_has_cls_token = False,
            text_pad_id = 0,
            text_rotary_pos_emb = False,
            text_causal_mask = False,
            text_eos_id = None,
            text_encode_without_mask = False,
            visual_enc_depth = 6,
            visual_heads = 8,
            visual_dim_head = 64,
            visual_image_size = 256,
            visual_patch_size = 32,
            visual_patch_dropout = 0.5,
            visual_has_cls_token = False,
            channels = 3,
            use_all_token_embeds = False,
            downsample_image_embeds = False,
            decoupled_contrastive_learning = False,
            extra_latent_projection = False,
            use_mlm = False,
            text_ssl_loss_weight = 0.05,
            use_visual_ssl = False,
            visual_ssl = None,
            visual_ssl_type = 'simsiam',
            visual_ssl_hidden_layer = -1,
            simclr_temperature = 0.1,
            image_ssl_loss_weight = 0.05,
            multiview_loss_weight = 0.1,
            checkpoint_during_training = False,
            **kwargs
    ):
        super().__init__()
        #assert use_all_token_embeds or (visual_has_cls_token or text_has_cls_token), 'CLS token must be included on both vision and text transformers if you are not using fine-grained contrastive learning loss'
        self.dtype=torch.float32
        # store some parameters for access

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.image_channels = channels
        self.image_size = visual_image_size

        # instantiate text transformer

        self.text_pad_id = text_pad_id
        self.text_has_cls_token = text_has_cls_token
        self.text_seq_len = text_seq_len

        self.text_encode_without_mask = text_encode_without_mask # whether to pass in text mask to text encoder

        self.text_causal_mask = text_causal_mask
        self.text_eos_id = text_eos_id

        assert not (text_causal_mask and not exists(text_eos_id)), 'text EOS token id must be given if using causal mask in text transformer'

        if exists(text_encoder):
            self.text_transformer = text_encoder
        else:
            self.text_transformer = TextTransformer(
                dim = dim_text,
                num_tokens = num_text_tokens + (1 if use_mlm else 0),
                max_seq_len = text_seq_len,
                depth = text_enc_depth,
                heads = text_heads,
                causal = text_causal_mask,
                dim_head = text_dim_head,
                rotary_pos_emb = text_rotary_pos_emb,
                checkpoint_during_training = checkpoint_during_training
            )

        # instantiate image transformer

        self.visual_has_cls_token = visual_has_cls_token

        if exists(image_encoder):
            self.visual_transformer = image_encoder
        else:
            self.visual_transformer = VisionTransformer(
                dim = dim_image,
                image_size = visual_image_size,
                patch_size = visual_patch_size,
                channels = channels,
                depth = visual_enc_depth,
                heads = visual_heads,
                dim_head = visual_dim_head,
                patch_dropout = visual_patch_dropout,
                checkpoint_during_training = checkpoint_during_training
            )

        # text ssl

        self.use_mlm = use_mlm
        self.text_ssl_loss_weight = text_ssl_loss_weight if use_mlm else 0

        if use_mlm:
            mlm_kwargs, kwargs = groupby_prefix_and_trim('mlm_', kwargs)
            self.mlm = MLM(
                self.text_transformer,
                dim = dim_text,
                num_tokens = num_text_tokens,
                **mlm_kwargs
            )

        # image ssl

        self.use_visual_ssl = use_visual_ssl or exists(visual_ssl)
        self.image_ssl_loss_weight = image_ssl_loss_weight if use_visual_ssl else 0

        if self.use_visual_ssl:
            if exists(visual_ssl):
                self.visual_ssl = visual_ssl

            elif use_visual_ssl:
                if visual_ssl_type == 'simsiam':
                    ssl_type = partial(SimSiam, channels = channels)
                elif visual_ssl_type == 'simclr':
                    ssl_type = partial(SimCLR, temperature = simclr_temperature, channels = channels)
                else:
                    raise ValueError(f'unknown visual_ssl_type')

                self.visual_ssl = ssl_type(
                    self.visual_transformer,
                    image_size = visual_image_size,
                    hidden_layer = visual_ssl_hidden_layer
                )

        # text latent projection

        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        # image latent projection for the CT

        if downsample_image_embeds: # by default this is false: potentially due to lower performance with lower resoultion image
            #assert use_all_token_embeds, 'must be using all token embeds for contrastive learning in order to downsampling'
            dim_conv=512
            self.to_visual_latent = nn.Sequential(
                RearrangeImage(),
                nn.Conv3d(dim_conv, dim_conv, 4, stride = 2, padding = 1, bias = False, groups = dim_conv),
                nn.Conv3d(dim_conv, dim_latent, 1),
                Rearrange('b c h w z -> b (h w z c)'),
                nn.Linear(dim_image, dim_latent, bias = False)
            )
        else:
            self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        # temperature

        self.temperature = nn.Parameter(torch.tensor(0.07))
        print(f'logit temperature {self.temperature}')

        # from https://arxiv.org/abs/2111.07783 (FILIP paper)
        self.use_all_token_embeds = use_all_token_embeds

        # proposed in https://arxiv.org/abs/2110.06848 (DCL) and https://arxiv.org/abs/2110.11316 (CLOOB)
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        # proposed in https://arxiv.org/abs/2110.11316 (CLOOB)
        self.extra_latent_projection = extra_latent_projection

        self.to_text_latent_extra = copy.deepcopy(self.to_text_latent)

        self.to_visual_latent_extra = copy.deepcopy(self.to_visual_latent)

        self.multiview_loss_weight = multiview_loss_weight

        # self.tokenizer= tokenizer if tokenizer else BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        warnings.filterwarnings('ignore')
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        self.load_state_dict(pt, strict=True)

    # def tokenize(self, prompt):
    #     text_tokens=self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(torch.cuda)
    #     return text_tokens
    def token_embedding(self,input_ids):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        if hasattr(self.text_transformer.embeddings, "token_type_ids"):
            print("hahatrue")

        buffered_token_type_ids = self.text_transformer.embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        token_type_ids = buffered_token_type_ids_expanded
        text_embeddings = self.text_transformer.embeddings(input_ids = input_ids, token_type_ids = token_type_ids)
        return text_embeddings

    def forward(
            self,
            text,
            image,
            device,
            return_loss = False,
            return_encodings = False,
            return_latents = False,
            freeze_image_encoder = False,   # image encoder is not trained if this is set to True, proposed by LiT paper
            freeze_text_encoder = False,    # text encoder is not trained if this is set to True
            text_to_image = True,           # in the case the extra projection is turned on, would return different similarity values depending on modality directionality
            aug_text = None,                # augmented text (for multiview)
            aug_image = None                # augmented image (for multiview)
    ):
        b, device = text.input_ids.shape[0], device

        # derive text mask

        text_mask =text.attention_mask

        # ssl

        text_ssl_loss = 0
        image_ssl_loss = 0

        if return_loss:
            #print("-----------")
            #print(text.input_ids.shape)
            #print(text.attention_mask.shape)
            #print("------------")
            text_ssl_loss = self.mlm(text.input_ids, attention_mask = text.attention_mask) if self.use_mlm else 0
            image_ssl_loss = self.visual_ssl(image) if self.use_visual_ssl else 0

        # concat augmented texts and images and do some asserts

        num_batch_texts = num_batch_images = 1

        if exists(aug_text):
            aug_text = cast_tuple(aug_text)
            assert all(map(lambda t: t.shape == text.shape, aug_text))
            num_batch_texts = len(aug_text) + 1

            aug_text = torch.cat(aug_text, dim = 0)

            aug_text_mask = aug_text != self.text_pad_id

            text_mask = torch.cat((text_mask, aug_text_mask), dim = 0)
            text = torch.cat((text, aug_text), dim = 0)

        if exists(aug_image):
            aug_image = cast_tuple(aug_image)
            assert all(map(lambda i: i.shape == image.shape, aug_image))
            num_batch_images = len(aug_image) + 1

            aug_image = torch.cat(aug_image, dim = 0)

            image = torch.cat((image, aug_image), dim = 0)

        is_multiview = (num_batch_texts > 1 or num_batch_images > 1)
        #assert not (return_loss and not self.training), 'loss cannot be used if not training'
        assert not (not return_loss and is_multiview), 'do not pass in augmented texts or images if not training'
        assert not (self.multiview_loss_weight == 0 and is_multiview), 'multiview loss weight cannot be 0 if augmented text or images passed in'

        # get encoded text

        text_args = (text.input_ids,text.attention_mask)

        if not self.text_encode_without_mask:
            text_args = (*text_args, text_mask)


        text_embeddings = self.text_transformer(text.input_ids, attention_mask = text.attention_mask )
        enc_text = text_embeddings[0] # global view of the text embeddings

        # depending on whether text is using causal mask, post process, moving eos token to the first position

        if self.text_causal_mask:
            eos_text_mask = (text == self.text_eos_id)
            assert torch.all(torch.any(eos_text_mask, dim = -1)), f'some of the text rows does not have the eos id {self.text_eos_id}'

            text_len = text.shape[-1]
            eos_indices = eos_text_mask.float().argmax(dim = -1, keepdim = True)

            eos_text_mask = torch.zeros_like(eos_text_mask).scatter(1, eos_indices, 1.).bool()
            eos_text_mask = rearrange(eos_text_mask, '... -> ... 1')

            eos_tokens = enc_text.masked_select(eos_text_mask)
            rest_tokens = enc_text.masked_select(~eos_text_mask)

            eos_tokens = rearrange(eos_tokens, '(b d) -> b 1 d', b = b)
            rest_tokens = rearrange(rest_tokens, '(b n d) -> b n d', b = b, n = text_len - 1)
            enc_text = torch.cat((eos_tokens, rest_tokens), dim = 1) #NOTE: concatenate in the seq dimension

        # whether to train image encoder, in the case that the image net was pretrained as recommended in LiT

        """enc_image = model_forward_with_context(
            fn = self.visual_transformer,
            args = (image,),
            freeze = freeze_image_encoder
        )"""

        enc_image= self.visual_transformer(image, return_encoded_tokens=True)

        #print("This is visual encoding")
        global h_r, w_r, z_r
        h_r, w_r, z_r = enc_image.shape[1], enc_image.shape[2], enc_image.shape[3]

        #enc_image, max_indices = torch.max(enc_image, dim=1)
        enc_image_send = enc_image

        enc_image = torch.mean(enc_image, dim=1)

        #kernel_size = (enc_image.size(1), enc_image.size(2), enc_image.size(3))

        #enc_image = enc_image.permute(0,4,1,2,3)
        # Perform max pooling over dimensions 1, 2, and 3
        #enc_image = F.max_pool3d(enc_image, kernel_size=kernel_size)

        #enc_image = enc_image.permute(0,2,3,4,1)

        #print(enc_image.shape, flush=True)
        #enc_image = enc_image[:,0,:]
        #print(enc_image.shape, flush=True)
        # print("test all pooling")
    
        # make the feature of the ct image in vector form batch x (h w z c)
        enc_image = enc_image.view(enc_image.shape[0], -1) # global view for one image and we have batch number of images

       # print(enc_image.shape, flush=True)

        # early return of encodings, if needed (for DALL-E2)

        if return_encodings:
            return enc_text, enc_image

        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only
        if self.use_all_token_embeds:
            assert enc_text.ndim == 3, 'encoded text must have 3 dimensions (batch, seq, features)'
            assert enc_image.ndim == 3, 'encoded image must have 3 dimensions (batch, seq [height x width], features)'
            text_embeds = enc_text[:, 1:] if self.text_has_cls_token else enc_text # get rid of the text global token
            image_embeds = enc_image[:, 1:] if self.visual_has_cls_token else enc_image # get rid of the visual global token
        else:
            # the [:,:] retains the same shape in this case
            text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
            image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image

        ## project to latents for both the ct image and the text modality
        #text_embeds = text_embeds.view(text_embeds.shape[0], -1)
        text_embeds = text_embeds[:,0,:]  # NOTE: Take the `[CLS]` token from the seq dimension

        #text_embeds = torch.mean(text_embeds, dim=1)
        text_latents = self.to_text_latent(text_embeds) #NOTE bxd
        image_latents = self.to_visual_latent(image_embeds) #NOTE bxd

        # normalize the features for both text and image
        text_latents, image_latents = map(l2norm, (text_latents, image_latents))

        # calculate another set of latents for image to text (vs text to image)
        # proposed by CLOOB

        text_latents_extra, image_latents_extra = text_latents, image_latents
        if self.extra_latent_projection:
            text_latents_extra = self.to_text_latent_extra(text_embeds)
            image_latents_extra = self.to_visual_latent_extra(image_embeds)
            text_latents_extra, image_latents_extra = map(l2norm, (text_latents_extra, image_latents_extra))

        # whether to early return latents NOTE: without computing the loss

        if return_latents:
            if self.extra_latent_projection:
                return text_latents, image_latents, text_latents_extra, image_latents_extra
            return text_latents, image_latents, enc_image_send

        # get temperature

        temp = self.temperature.exp()

        # early return, if needed

        #NOTE: direct return the similarity score
        if not return_loss and self.use_all_token_embeds: # Fine-grained token-to-token similarity
            einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (text_latents, image_latents)
            return einsum('b d, b i d -> b t i', *einsum_args) * temp

        if not return_loss and not self.use_all_token_embeds: # Global text-to-image similarity
            einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (text_latents, image_latents)
            return einsum('b d, b d -> b', *einsum_args) * temp # NOTE: check the previous implementation, you see that the image feature is of the shape bxd under this condition

        # split out multiview dimension for text and images
        # NOTE: for not multi view, m = 1
        text_latents = rearrange(text_latents, '(m b) ... -> m b ...', m = num_batch_texts) #NOTE: 1xbxd
        image_latents = rearrange(image_latents, '(m b) ... -> m b ...', m = num_batch_images) #NOTE: 1xbxd

        if self.extra_latent_projection:
            text_latents_extra = rearrange(text_latents_extra, '(m b) ... -> m b ...', m = num_batch_texts)
            image_latents_extra = rearrange(image_latents_extra, '(m b) ... -> m b ...', m = num_batch_images)

        # contrastive loss

        """
        m - num batches of text (for multiview)
        n - num batches of images (for multiview)
        x - batches of text
        y - batches of images
        t - sequence dimension along text tokens
        i - sequence dimension along image tokens
        """

        if self.use_all_token_embeds:
            # fine-grained CLIP logic
            sim_text_to_image = einsum('m x t d, n y i d -> m n x y t i', text_latents, image_latents) * temp

            sim_image_to_text = sim_text_to_image
            if self.extra_latent_projection:
                sim_image_to_text = einsum('m x t d, n y i d -> m n x y t i', text_latents_extra, image_latents_extra) * temp

            text_to_image = reduce(sim_text_to_image, '... t i -> ... t', 'max')
            text_to_image_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t', m = num_batch_texts).bool()
            text_to_image = masked_mean(text_to_image, text_to_image_mask, dim = -1)

            image_to_text_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t 1', m = num_batch_texts).bool()
            masked_sim = sim_image_to_text.masked_fill(~image_to_text_mask, max_neg_value(sim_image_to_text.dtype))
            image_to_text = reduce(reduce(masked_sim, '... t i -> ... i', 'max'), '... i -> ...', 'mean')
        else:
            text_to_image = einsum('m t d, n i d -> m n t i', text_latents, image_latents) * temp # NOTE: one axis of the similarity matrix, for m=n=1: 1x1xbxb
            image_to_text = rearrange(text_to_image, '... t i -> ... i t') # NOTE: the other axis of the similarity matrix, for m=n=1: 1x1xbxb
            # NOTE: this compute the fine-grain token level similarity for each combination of views (mxn)
            if self.extra_latent_projection:
                image_to_text = einsum('m t d, n i d -> m n i t', text_latents_extra, image_latents_extra) * temp

        # calculate loss

        text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
        image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')


        # exponentiate NOTE: expoential everything
        text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))

        # numerators NOTE: pick up the digonal terms in the matrix
        text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp)) # for m=n=1, get the diagonal terms from matrix of shape 1x1xbxb

        # denominator

        if self.decoupled_contrastive_learning:
            pos_mask = torch.eye(b, device = device, dtype = torch.bool)
            text_to_image_exp, image_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (text_to_image_exp, image_to_text_exp))

        #NOTE: this sum up the each axis of the similarity matrix
        # (m, n, t) and (m,n, i), if m = n = 1, then (1,1, b), (1,1, b)
        text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim = -1), (text_to_image_exp, image_to_text_exp)) #NOTE: in 1x1xbx1

        # loss
        # NOTE: (1,1, b) + (1,1, b (denominator of the CL term for each image in the batch)) operation then take the average of it
        # NOTE: the mean operation mainly to normalize in the batch direction, agnostic to the batch size essentially.
        text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean(dim = -1) # t->i log(CL)
        image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean(dim = -1) # i->t log(CL)

        # calculate CL loss

        cl_losses = (text_to_image_loss + image_to_text_loss) / 2 #NOTE: symmetry loss text->image and image->text

        # get main CL loss vs multiview CL losses

        cl_loss, multiview_cl_loss = cl_losses[0], cl_losses[1:]

        # if no augmented text or images passed in, multiview loss weight is 0

        multiview_loss_weight = self.multiview_loss_weight if is_multiview else 0

        # calculate weights

        cl_loss_weight = 1 - (self.text_ssl_loss_weight + self.image_ssl_loss_weight + multiview_loss_weight)

        loss = (cl_loss * cl_loss_weight) \
               + (text_ssl_loss * self.text_ssl_loss_weight) \
               + (image_ssl_loss * self.image_ssl_loss_weight)

        # add multiview CL loss with weight

        if is_multiview:
            loss = loss + multiview_cl_loss.mean() * multiview_loss_weight

        return loss

class CTCLIPwithXray(nn.Module):
    def __init__(
            self,
            *,
            image_encoder = None,
            text_encoder = None,
            tokenizer = None,
            xray_model_type = 'cxr_clip_swin', # any vit based
            dim_text = 512,
            dim_image = 512,
            dim_xray = 512,
            dim_latent = 512,
            num_text_tokens = 28897,
            text_enc_depth = 6,
            text_seq_len = 256,
            text_heads = 8,
            text_dim_head = 64,
            text_has_cls_token = False,
            text_pad_id = 0,
            text_rotary_pos_emb = False,
            text_causal_mask = False,
            text_eos_id = None,
            text_encode_without_mask = False,
            visual_enc_depth = 6,
            visual_heads = 8,
            visual_dim_head = 64,
            visual_image_size = 256,
            visual_patch_size = 32,
            visual_patch_dropout = 0.5,
            visual_has_cls_token = False,
            channels = 3,
            use_all_token_embeds = False,
            downsample_image_embeds = False,
            decoupled_contrastive_learning = False,
            extra_latent_projection = False,
            use_mlm = False,
            text_ssl_loss_weight = 0.05,
            use_visual_ssl = False,
            visual_ssl = None,
            visual_ssl_type = 'simsiam',
            visual_ssl_hidden_layer = -1,
            simclr_temperature = 0.1,
            image_ssl_loss_weight = 0.05,
            multiview_loss_weight = 0.1,
            checkpoint_during_training = False,
            cfg=None,
            auto_load_pretrained_weights=True,
            freeze_xray_pretrained_weights=True,
            **kwargs
    ):
        super().__init__()
        self.CTCLIP = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            tokenizer = tokenizer,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
            num_text_tokens=num_text_tokens,
            text_enc_depth=text_enc_depth,
            text_seq_len=text_seq_len,
            text_heads=text_heads,
            text_dim_head=text_dim_head,
            text_has_cls_token=text_has_cls_token,
            text_pad_id=text_pad_id,
            text_rotary_pos_emb=text_rotary_pos_emb,
            text_causal_mask=text_causal_mask,
            text_eos_id=text_eos_id,
            text_encode_without_mask=text_encode_without_mask,
            visual_enc_depth=visual_enc_depth,
            visual_heads=visual_heads,
            visual_dim_head=visual_dim_head,
            visual_image_size=visual_image_size,
            visual_patch_size=visual_patch_size,
            visual_patch_dropout=visual_patch_dropout,
            visual_has_cls_token=visual_has_cls_token,
            channels=channels,
            use_all_token_embeds=use_all_token_embeds,
            downsample_image_embeds=downsample_image_embeds,
            decoupled_contrastive_learning=decoupled_contrastive_learning,
            extra_latent_projection=extra_latent_projection,
            use_mlm=use_mlm,
            text_ssl_loss_weight=text_ssl_loss_weight,
            use_visual_ssl=use_visual_ssl,
            visual_ssl=visual_ssl,
            visual_ssl_type=visual_ssl_type,
            visual_ssl_hidden_layer=visual_ssl_hidden_layer,
            simclr_temperature=simclr_temperature,
            image_ssl_loss_weight=image_ssl_loss_weight,
            multiview_loss_weight=multiview_loss_weight,
            checkpoint_during_training=checkpoint_during_training,
            **kwargs
        )

        #NOTE: with the xray encoder
        self.cfg = cfg
        self.xray_model_type = xray_model_type
        if xray_model_type == 'ct_clip':
            self.xray_encoder = None
            self.to_xray_latent = None
        elif xray_model_type == 'cxr_clip_swin': # default options.
            # load the plain image encoder
            self.xray_encoder = load_cxr_clip_image_encoder(cfg["swin"]["image_encoder"])
            self.to_xray_latent = nn.Linear(dim_xray, dim_latent, bias = False)

            if auto_load_pretrained_weights:
                # load the cxr_clip pretrained weights to the swin encoder as well as the to_xray_latent prejection layer
                ckpt_file_name = 'swint_mcc'
                self.load_cxr_clip_xray_encoder(
                    '/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/cxr_clip/{}.tar'.format(ckpt_file_name), # cxr-clip pretrained
                    freeze_weights=freeze_xray_pretrained_weights
                )
                print('loaded xray encoder from cxr_clip SWIN')
            else:
                print('NOT LOADING ANY MEDICAL RELATED PRETRAINED WEIGHTS')

        elif xray_model_type == 'cxr_clip_resnet':
            # load the plain image encoder
            self.xray_encoder = load_cxr_clip_image_encoder(cfg["resnet"]["image_encoder"])
            self.to_xray_latent = nn.Linear(dim_xray, dim_latent, bias = False)

            if auto_load_pretrained_weights:
                # load the cxr_clip pretrained weights to the resnet encoder as well as the to_xray_latent prejection layer
                ckpt_file_name = 'r50_mcc'
                self.load_cxr_clip_xray_encoder(
                    '/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/cxr_clip/{}.tar'.format(ckpt_file_name), # cxr-clip pretrained
                    freeze_weights=freeze_xray_pretrained_weights
                )
                print('loaded xray encoder from cxr_clip RESNET')
            else:
                print('NOT LOADING ANY MEDICAL RELATED PRETRAINED WEIGHTS')

        # NOTE: the rest of the baseline always load the pretrained model including the projection layer
        elif xray_model_type == 'medclip_resnet':
            medclip_vision_encoder = MedCLIPVisionModel(MedCLIPVisionModelResNet, checkpoint='/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/medclip/resnet' if auto_load_pretrained_weights else None)

            self.to_xray_latent = copy.deepcopy(medclip_vision_encoder.vision_model.model.fc)
            medclip_vision_encoder.vision_model.model.fc = nn.Identity() # delete the fc layer
            self.xray_encoder = copy.deepcopy(medclip_vision_encoder.vision_model.model)
            if freeze_xray_pretrained_weights:
                self.freeze_xray_encoder_weights()
            print('loaded xray encoder from medclip_resnet')
            
        elif xray_model_type == 'medclip_vit':
            medclip_vision_encoder = MedCLIPVisionModel(MedCLIPVisionModelViT, checkpoint='/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/medclip/vit' if auto_load_pretrained_weights else None)
            self.to_xray_latent = copy.deepcopy(medclip_vision_encoder.vision_model.projection_head)
            self.xray_encoder = copy.deepcopy(medclip_vision_encoder.vision_model.model)
            if freeze_xray_pretrained_weights:
                self.freeze_xray_encoder_weights()
            print('loaded xray encoder from medclip_vit')
            
        elif xray_model_type == 'gloria_densenet':
            gloria_vision_encoder = GloRIaVisionModel(GloRIaVisionModelDenseNet, checkpoint='/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/gloria/pytorch_version/chexpert_densenet121_torch_version.pth')

            self.to_xray_latent = copy.deepcopy(gloria_vision_encoder.vision_model.model.classifier) # the head need to be used for linear probing
            gloria_vision_encoder.vision_model.model.classifier = nn.Identity() # delete the classifier layer
            self.xray_encoder = copy.deepcopy(gloria_vision_encoder.vision_model.model)
            print('loaded xray encoder from gloria_densenet')

        elif xray_model_type == 'gloria_resnet':
            gloria_vision_encoder = GloRIaVisionModel(GloRIaVisionModelResNet, checkpoint='/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/gloria/pytorch_version/chexpert_resnet50_torch_version.pth')

            self.to_xray_latent = copy.deepcopy(gloria_vision_encoder.vision_model.model.fc) # the head need to be used for linear probing
            gloria_vision_encoder.vision_model.model.fc = nn.Identity() # delete the fc layer
            self.xray_encoder = copy.deepcopy(gloria_vision_encoder.vision_model.model)
            print('loaded xray encoder from gloria_resnet')
        else: 
            # our pretrained model
            ckpt_name = xray_model_type
            self.xray_encoder = load_cxr_clip_image_encoder(
                cfg["swin"]["image_encoder"] if 'swin' in ckpt_name.lower() else cfg["resnet"]["image_encoder"]
            )
            self.to_xray_latent = nn.Linear(dim_xray, dim_latent, bias = False)

            if auto_load_pretrained_weights:
                # ckpt_name='modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch'
                #NOTE: weights for projection layer and the encoder body will be loaded, guaranteed by strict=True
                self.load_our_pretrained_weights(f'/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/{ckpt_name}.pt', freeze_weights=freeze_xray_pretrained_weights)
                print(f'Loaded custom pretrained weights from {ckpt_name}')
            else:
                print('NOT LOADING ANY MEDICAL RELATED PRETRAINED WEIGHTS')


    def forward(
            self,
            text,
            image,
            xray,
            device,
            text_cl_weight = 1.0,
            ct_cl_weight = 1.0,
            is_text_latent_input = True, # for triplet modal training, by default it is 
            is_image_latent_input = True, # for triplet modal training, by default it is 
            return_logits_only = False,
    ):
        # print(f'text cl loss weight {text_cl_weight}, ct cl loss weight {ct_cl_weight}')
        num_batch_texts = num_batch_images = 1
        if not is_text_latent_input:
            """
            NOTE: the following implementation mostly adapted from the parent class without 
                    multiview,
                    casual mask,
                    use_all_token_embeds,
                    extra_latent_projection,
                    decoupled_contrastive_learning
            """
            b, device = text.input_ids.shape[0], device # batch size, device
            
            text_embeddings = self.CTCLIP.text_transformer(text.input_ids, attention_mask = text.attention_mask)
            enc_text = text_embeddings[0] # [0] are the tokens feature, [1] is the pooled features

            # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only
            text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
            
            # project to latents for the text modality
            text_embeds = text_embeds[:,0,:]  # NOTE: Take the `[CLS]` token from the seq dimension
            text_latents = self.CTCLIP.to_text_latent(text_embeds) #NOTE bxd

            # normalize the features for both text and image
            text_latents = l2norm(text_latents)
        else:
            # inputs are embedding itself
            text_latents = text
        
        if not is_image_latent_input:
            enc_image= self.CTCLIP.visual_transformer(image, return_encoded_tokens=True)
            global h_r, w_r, z_r
            h_r, w_r, z_r = enc_image.shape[1], enc_image.shape[2], enc_image.shape[3]

            # make the feature of the ct image in vector form batch x (h w z c)
            enc_image = torch.mean(enc_image, dim=1) # pool the patch features
            enc_image = enc_image.view(enc_image.shape[0], -1) # global view for each vol in a batch

            # project to latents for the ct image
            image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image
            image_latents = self.CTCLIP.to_visual_latent(image_embeds) #NOTE bxd

            # normalize the features for the image
            image_latents = l2norm(image_latents)
        else:
            image_latents = image

        # always extract xray feature representation
        xray_latents = self.get_xray_latents(xray)

        # get temperature
        temp = self.CTCLIP.temperature.exp()

        # split out multiview dimension for text and images
        text_latents = rearrange(text_latents, '(m b) ... -> m b ...', m = num_batch_texts) #NOTE: 1xbxd
        image_latents = rearrange(image_latents, '(m b) ... -> m b ...', m = num_batch_images) #NOTE: 1xbxd
        xray_latents = rearrange(xray_latents, '(m b) ... -> m b ...', m = num_batch_images) #NOTE: 1xbxd

        if return_logits_only:
            logits = einsum('m t d, n i d -> m n t i', text_latents, xray_latents) * temp # compute similarity matrix
            return logits.squeeze()
        """
        NOTE: CL between image and xray and CL between text and xray
        """
        cl_text_to_xray = self.cl_loss(text_latents, xray_latents, temp)
        cl_img_to_xray = self.cl_loss(image_latents, xray_latents, temp)
        loss = text_cl_weight*cl_text_to_xray + ct_cl_weight*cl_img_to_xray
      
        return loss

    def get_xray_latents(self, xray):
        # always extract xray feature representation
        enc_xray = self.xray_encoder(xray)

        if 'resnet' in self.xray_model_type.lower():
            enc_xray = enc_xray.view(enc_xray.shape[0], 1, -1)
        elif self.xray_model_type == 'medclip_vit':
            enc_xray = enc_xray.last_hidden_state

        enc_xray = torch.mean(enc_xray, dim=1) # pool the patch features [batch size, patches, features] => [batch size, features]
        enc_xray = enc_xray.view(enc_xray.shape[0], -1) # global view for each xray in a batch of shape [batch size, features]
        xray_embeds = enc_xray[:, :] if enc_xray.ndim == 3 else enc_xray
        xray_latents = self.to_xray_latent(xray_embeds)
        xray_latents = l2norm(xray_latents)

        return xray_latents

    def get_report_latent(self, text):
        text_embeddings = self.CTCLIP.text_transformer(text.input_ids, attention_mask = text.attention_mask)
        enc_text = text_embeddings[0] # [0] are the tokens feature, [1] is the pooled features

        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only
        text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
        
        # project to latents for the text modality
        text_embeds = text_embeds[:,0,:]  # NOTE: Take the `[CLS]` token from the seq dimension
        text_latents = self.CTCLIP.to_text_latent(text_embeds) #NOTE bxd

        # normalize the features for both text and image
        text_latents = l2norm(text_latents)
        return text_latents

    def cl_loss(self, m1_latent, m2_latent, temp):
        """
        compute the contrastive loss between the latent features of m1 and m2 modality
        """
        
        m1_to_m2 = einsum('m t d, n i d -> m n t i', m1_latent, m2_latent) * temp # compute similarity matrix
        m2_to_m1 = rearrange(m1_to_m2, '... t i -> ... i t') # another axis

        ## calculate loss
        m1_to_m2 = rearrange(m1_to_m2, 'm n ... -> (m n) ...')
        m2_to_m1 = rearrange(m2_to_m1, 'm n ... -> (m n) ...')

        # exponentiate NOTE: expoential everything
        m1_to_m2_exp, m2_to_m1_exp = map(torch.exp, (m1_to_m2, m2_to_m1))

        # numerators NOTE: pick up the digonal terms in the matrix
        m1_to_m2_pos, m2_to_m1_pos = map(matrix_diag, (m1_to_m2_exp, m2_to_m1_exp)) # for m=n=1, get the diagonal terms from matrix of shape 1x1xbxb

        # denominator
        #NOTE: this sum up the each axis of the similarity matrix
        # (m, n, t) and (m,n, i), if m = n = 1, then (1,1, b), (1,1, b)
        m1_to_m2_denom, m2_to_m1_denom = map(lambda t: t.sum(dim = -1), (m1_to_m2_exp, m2_to_m1_exp)) #NOTE: in 1x1xbx1

        # loss
        m1_to_m2_loss = (-log(m1_to_m2_pos) + log(m1_to_m2_denom)).mean(dim = -1) # t->i log(CL)
        m2_to_m1_loss = (-log(m2_to_m1_pos) + log(m2_to_m1_denom)).mean(dim = -1) # i->t log(CL)

        # calculate CL loss
        loss = (m1_to_m2_loss + m2_to_m1_loss) / 2 #NOTE: symmetry loss m1->m2 and m2->m1

        return loss

    def load_our_pretrained_weights(self, weight_path, freeze_weights=True):
        """load the pretrained model from our own pretrained modified ct-clip model"""
        #NOTE: this is strict loading => promised the weights are loaded. including the latent projection layer.

        weights = torch.load(weight_path, weights_only=True)
        self.load_state_dict(weights, strict=True) # load everything that is previously saved

        if freeze_weights:
            # NOTE: this freezed everything including the the latent layer.
            print("    freezing weights in XRAY_CTCLIP")
            for param in self.parameters():
                param.requires_grad = False

    def load(self, ctclip_path, cxr_path):
        self.load_ctclip(ctclip_path, freeze_weights=True)
        if cxr_path: # if false, implies randomly initialized
            self.load_cxr_clip_xray_encoder(cxr_path, False)
        else:
            print('NOT loading the pretrained cxr_clip weights for the xray encoder')
    
    def load_ctclip(self, ctclip_path, freeze_weights=True):
        """this only load the CT-CLIP model (the original model from the CT-CLIP paper)"""
        warnings.filterwarnings('ignore')
        #NOTE: this is strict loading => promised the weights are loaded. including the latent projection layer.

        # load the pretrained model for the ctclip
        self.CTCLIP.load(ctclip_path)
        print('    finished loading the checkpoint for ct clip encoders')

        #NOTE: freeze the image and text backbones
        if freeze_weights:
            print("    freezing weights in CTCLIP")
            for param in self.CTCLIP.parameters():
                param.requires_grad = False
    

    def load_cxr_clip_xray_encoder(self, cxr_path, freeze_weights=False):
        """handle only loading the cxr_clip based xray encoder and its (not ours) projection layer only -- need special handling of the dictionary keys like below"""
        warnings.filterwarnings('ignore')
        ckpt = torch.load(cxr_path, map_location="cpu")

        # NOTE: the following only valid for swinT in the cxr_clip
        # [key for key in ckpt["model"].keys() if 'image_encoder' in key] # NOTE this is the way to check the keys
        saved_state_dict = ckpt["model"]
        new_state_dict = {}
        for key in saved_state_dict.keys():
            if 'image_encoder.' in key:
                new_state_dict[key.replace("image_encoder.", "xray_encoder.", 1)] = saved_state_dict[key]
            # note that during loading of the resnet model in cxr_clip, it removes the fc layer by default and thuse we need to add it back
            # check the CXRClip.forward method line 28 and line 79: https://github.dev/Soombit-ai/cxr-clip/blob/31188857cc5e892c22c731b176080eb5d4484cf2/cxrclip/model/clip.py#L79-L80
            if 'image_projection.projection.weight' in key:
                new_state_dict[key.replace("image_projection.projection.weight", "to_xray_latent.weight", 1)] = saved_state_dict[key]

        missing_keys, _ = self.load_state_dict(new_state_dict, strict=False)
        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(new_state_dict.keys())
        loaded_keys = ckpt_keys.intersection(model_keys) - set(missing_keys)

        # NOTE: this sanity check make sure the weights of the projection head is loaded. remember is it deleted in the custom RESNET class in cxr_clip_utils class
        assert (len(self.xray_encoder.state_dict().keys()) + 1) == len(loaded_keys)
        print(f'    finished loading the weights of the xray encoder from cxr_clip: {cxr_path}')

        #NOTE: freeze the image and text backbones
        if freeze_weights:
            self.freeze_xray_encoder_weights()
        
    def freeze_xray_encoder_weights(self):
        print("    freezing weights in XRay encoder including the projection layer")
        for param in self.xray_encoder.parameters():
            param.requires_grad = False
        for param in self.to_xray_latent.parameters():
            param.requires_grad = False