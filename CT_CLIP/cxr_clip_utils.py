from typing import Dict, Union, TypeVar
from omegaconf import DictConfig
import torch
from torch import nn
from torchvision.models.resnet import resnet50
from transformers import AutoConfig, AutoModel, SwinModel, ViTModel
import albumentations
import albumentations.pytorch.transforms
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import random
import os
from transformers import AutoConfig, AutoModel, BertModel
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.nn import Module

T = TypeVar("T", bound="Module")

class HuggingfaceTextEncoder(nn.Module):
    def __init__(
        self,
        name: str = "bert-base-uncased",
        vocab_size: int = None,
        pretrained: bool = True,
        gradient_checkpointing: bool = False,
        cache_dir: str = "~/.cache/huggingface/hub",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        if pretrained:
            self.text_encoder = AutoModel.from_pretrained(
                name,
                # vocab_size=vocab_size,
                ignore_mismatched_sizes=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        else:
            # initializing with a config file does not load the weights associated with the model
            model_config = AutoConfig.from_pretrained(
                name,
                # vocab_size=vocab_size,
                ignore_mismatched_sizes=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            if type(model_config).__name__ == "BertConfig":
                self.text_encoder = BertModel(model_config)
            else:
                raise NotImplementedError(f"Not support training from scratch : {type(model_config).__name__}")

        if gradient_checkpointing and self.text_encoder.supports_gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable()

        self.out_dim = self.text_encoder.config.hidden_size

    def forward(self, x):
        output = self.text_encoder(**x)
        return output["last_hidden_state"]  # (batch, seq_len, hidden_size)


class HuggingfaceImageEncoder(nn.Module):
    def __init__(
        self,
        name: str = "google/vit-base-patch16-224",
        pretrained: bool = True,
        gradient_checkpointing: bool = False,
        cache_dir: str = "~/.cache/huggingface/hub",
        model_type: str = "vit",
        local_files_only: bool = False,
    ):
        super().__init__()
        self.model_type = model_type
        if pretrained:
            if self.model_type == "swin":
                self.image_encoder = SwinModel.from_pretrained(
                    name,
                    cache_dir=cache_dir, 
                    local_files_only=local_files_only
                )
            else:
                self.image_encoder = AutoModel.from_pretrained(
                    name, add_pooling_layer=False, cache_dir=cache_dir, local_files_only=local_files_only
                )
        else:
            # initializing with a config file does not load the weights associated with the model
            model_config = AutoConfig.from_pretrained(name, cache_dir=cache_dir, local_files_only=local_files_only)
            if type(model_config).__name__ == "ViTConfig":
                self.image_encoder = ViTModel(model_config, add_pooling_layer=False)
            else:
                raise NotImplementedError(f"Not support training from scratch : {type(model_config).__name__}")

        if gradient_checkpointing and self.image_encoder.supports_gradient_checkpointing:
            self.image_encoder.gradient_checkpointing_enable()

        self.out_dim = self.image_encoder.config.hidden_size

    def forward(self, image):
        if self.model_type == "vit":
            output = self.image_encoder(pixel_values=image, interpolate_pos_encoding=True)
        elif self.model_type == "swin":
            output = self.image_encoder(pixel_values=image)
        return output["last_hidden_state"]  # (batch, seq_len, hidden_size)


class ResNet50(nn.Module):
    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.resnet = resnet50(pretrained=True)
        else:
            raise NotImplementedError(f"Not support training from scratch : {name}")

        self.out_dim = 2048
        del self.resnet.fc
        self.resnet = nn.SyncBatchNorm.convert_sync_batchnorm(self.resnet)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x
    
class MLPProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class LinearProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)

    def forward(self, x):
        return self.projection(x)

class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_class):
        super().__init__()
        self.classification_head = nn.Linear(feature_dim, num_class)

    def forward(self, x):
        return self.classification_head(x)
    

def transform_image(image_transforms, image: Union[Image.Image, np.ndarray], normalize="huggingface"):
    for tr in image_transforms:
        if isinstance(tr, albumentations.BasicTransform):
            image = np.array(image) if not isinstance(image, np.ndarray) else image
            image = tr(image=image)["image"]
        else:
            image = transforms.ToPILImage()(image) if not isinstance(image, Image.Image) else image
            image = tr(image)

    if normalize == "huggingface":
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(image)

    elif normalize == "imagenet":
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    else:
        raise KeyError(f"Not supported Normalize: {normalize}")

    return image

def load_transform(split: str = "train", transform_config: Dict = None):
    assert split in {"train", "valid", "test", "aug"}

    config = []
    if transform_config:
        if split in transform_config:
            config = transform_config[split]
    image_transforms = []

    for name in config:
        if hasattr(transforms, name):
            tr_ = getattr(transforms, name)
        else:
            tr_ = getattr(albumentations, name)
        tr = tr_(**config[name])
        image_transforms.append(tr)

    return image_transforms

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed


def convert_dictconfig_to_dict(cfg):
    if isinstance(cfg, DictConfig):
        return {k: convert_dictconfig_to_dict(v) for k, v in cfg.items()}
    else:
        return cfg
    

def load_image_encoder(config_image_encoder: Dict):
    if config_image_encoder["source"].lower() == "huggingface":
        cache_dir = config_image_encoder["cache_dir"] if "cache_dir" in config_image_encoder else "~/.cache/huggingface/hub"
        gradient_checkpointing = (
            config_image_encoder["gradient_checkpointing"] if "gradient_checkpointing" in config_image_encoder else False
        )
        model_type = config_image_encoder["model_type"] if "model_type" in config_image_encoder else "vit"
        _image_encoder = HuggingfaceImageEncoder( #NOTE: potentially for the swin vision transformer model
            name=config_image_encoder["name"],
            pretrained=config_image_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            model_type=model_type,
            local_files_only=os.path.exists(os.path.join(cache_dir, f'models--{config_image_encoder["name"].replace("/", "--")}')),
        )
    elif config_image_encoder["name"] == "resnet":
        _image_encoder = ResNet50()

    else:
        raise KeyError(f"Not supported image encoder: {config_image_encoder}")
    return _image_encoder


def load_text_encoder(config_text_encoder: Dict, vocab_size: int):
    if config_text_encoder["source"].lower() == "huggingface":
        cache_dir = config_text_encoder["cache_dir"]
        gradient_checkpointing = config_text_encoder["gradient_checkpointing"]
        _text_encoder = HuggingfaceTextEncoder(
            name=config_text_encoder["name"],
            vocab_size=vocab_size,
            pretrained=config_text_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(os.path.join(cache_dir, f'models--{config_text_encoder["name"].replace("/", "--")}')),
            trust_remote_code=config_text_encoder["trust_remote_code"],
        )
    else:
        raise KeyError(f"Not supported text encoder: {config_text_encoder}")
    return _text_encoder


def load_projection_head(embedding_dim: int, config_projection_head: Dict):
    if config_projection_head["name"].lower() == "mlp":
        projection_head = MLPProjectionHead(
            embedding_dim=embedding_dim, projection_dim=config_projection_head["proj_dim"], dropout=config_projection_head["dropout"]
        )
    elif config_projection_head["name"].lower() == "linear":
        projection_head = LinearProjectionHead(embedding_dim=embedding_dim, projection_dim=config_projection_head["proj_dim"])
    else:
        raise KeyError(f"Not supported text encoder: {config_projection_head}")
    return projection_head


def load_image_classifier(config_image_classifier: Dict, feature_dim: int):
    if config_image_classifier["name"].lower() == "linear":
        _image_classifier = LinearClassifier(feature_dim=feature_dim, num_class=config_image_classifier["n_class"])
    else:
        raise KeyError(f"Not supported image classifier: {config_image_classifier}")

    return _image_classifier


def build_model(model_config: Dict, loss_config: Dict, tokenizer: PreTrainedTokenizer = None) -> nn.Module:
    if model_config["name"].lower() == "clip_custom": #NOTE: look for clip_custom to find the config file
        model = CXRClip(model_config, loss_config, tokenizer)
    elif model_config["name"].lower() == "finetune_classification":
        #NOTE this one only train the linear classifier
        model_type = model_config["image_encoder"]["model_type"] if "model_type" in model_config["image_encoder"] else "vit"
        model = CXRClassification(model_config, model_type)
    else:
        raise KeyError(f"Not supported model: {model_config['name']}")
    return model


class CXRClip(nn.Module):
    #NOTE: main CXR_CLIP model
    def __init__(self, model_config: Dict, all_loss_config: Dict, tokenizer: PreTrainedTokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_encoder = load_image_encoder(model_config["image_encoder"])
        self.text_encoder = load_text_encoder(model_config["text_encoder"], vocab_size=tokenizer.vocab_size)
        self.text_pooling = model_config["text_encoder"]["pooling"]

        self.model_config = model_config
        self.loss_config = {k: v for k, v in all_loss_config.items()}

        self.projection = "projection_head" in model_config

        if self.projection:
            self.image_projection = load_projection_head(
                embedding_dim=self.image_encoder.out_dim, config_projection_head=model_config["projection_head"]
            )
            self.text_projection = load_projection_head(
                embedding_dim=self.text_encoder.out_dim, config_projection_head=model_config["projection_head"]
            )
        else:
            assert (
                self.image_encoder.out_dim == self.text_encoder.out_dim
            ), "Without 'projection_head', embedding_dim of the image and text encoder must be the same."

        self.temperature = model_config["temperature"] if "temperature" in model_config else None
        if self.temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        else:
            self.logit_scale = torch.tensor(1, dtype=torch.float32)
            # log.warning("[CXRCLIP] missing temperature scaling factor")
            print('[CXRCLIP] missing temperature scaling factor')

    def encode_image(self, image):
        image_features = self.image_encoder(image)

        if self.model_config["image_encoder"]["name"] == "resnet":
            return image_features
        else:
            # get [CLS] token for global representation (only for vision transformer)
            global_features = image_features[:, 0]
            return global_features

    def encode_text(self, text_tokens):
        text_features = self.text_encoder(text_tokens)

        if self.text_pooling == "eos":
            # take features from the eot embedding (eos_token is the highest number in each sequence)
            eos_token_indices = text_tokens["attention_mask"].sum(dim=-1) - 1
            text_features = text_features[torch.arange(text_features.shape[0]), eos_token_indices]
        elif self.text_pooling == "bos":
            text_features = text_features[:, 0]
        elif self.text_pooling == "mean":
            input_mask_expanded = text_tokens["attention_mask"].unsqueeze(axis=-1).expand(text_features.size()).float()
            text_features = torch.sum(text_features * input_mask_expanded, axis=1) / torch.clamp(input_mask_expanded.sum(axis=1), min=1e-9)
        else:
            raise NotImplementedError("Not supported pooling method : %s", self.text_pooling)

        return text_features

    def forward(self, batch, device=None):
        device = batch["images"].device if device is None else device
        # get image and text features
        image_features_g = self.encode_image(batch["images"].to(device))
        text_features_g = self.encode_text(batch["text_tokens"].to(device))

        image_embeddings = self.image_projection(image_features_g) if self.projection else image_features_g
        text_embeddings = self.text_projection(text_features_g) if self.projection else text_features_g

        # normalize features
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        # labels
        labels = torch.arange(image_embeddings.shape[0], device=device)

        out = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "labels": labels,
            "logit_scale": self.logit_scale.exp(),
        }

        text_features_g2 = self.encode_text(batch["text_tokens2"].to(device))
        text_embeddings2 = self.text_projection(text_features_g2) if self.projection else text_features_g
        text_embeddings2 = text_embeddings2 / text_embeddings2.norm(dim=1, keepdim=True)
        out["text_embeddings2"] = text_embeddings2

        image_view_encode = self.encode_image(batch["image_views"].to(device))
        image_view_embeddings = self.image_projection(image_view_encode) if self.projection else image_view_encode
        image_view_embeddings = image_view_embeddings / image_view_embeddings.norm(dim=1, keepdim=True)
        out["image_view_embeddings"] = image_view_embeddings

        return out


class CXRClassification(nn.Module):
    def __init__(self, model_config: Dict, model_type: str = "vit"):
        super().__init__()
        self.model_type = model_type
        self.model_config = model_config
        if model_config["load_backbone_weights"] is None:
            self.image_encoder = load_image_encoder(model_config["image_encoder"])
        else:
            #NOTE: sample code to load the backbone weights
            # log.info("    loading pre-trained image encoder for fine-tuning")
            print('    loading pre-trained image encoder for fine-tuning')
            if not os.path.isfile(model_config["load_backbone_weights"]):
                raise ValueError(f"Cannot find a weight file: {model_config['load_backbone_weights']}")
            ckpt = torch.load(model_config["load_backbone_weights"], map_location="cpu")
            print(ckpt["config"]["model"]["image_encoder"])
            self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])
            image_encoder_weights = {}
            for k in ckpt["model"].keys():
                if k.startswith("image_encoder."):
                    image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
            self.image_encoder.load_state_dict(image_encoder_weights, strict=True)

        if model_config["freeze_backbone_weights"]:
            # log.info("    freezing image encoder to not be trained")
            print('    freezing image encoder to not be trained')
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.classifier = load_image_classifier(model_config["classifier"]["config"], self.image_encoder.out_dim)

    def encode_image(self, image):
        image_features = self.image_encoder(image)

        if self.model_config["image_encoder"]["name"] == "resnet":
            return image_features
        else:
            # get [CLS] token for global representation (only for vision transformer)
            global_features = image_features[:, 0]
            return global_features

    def train(self: T, mode: bool = True) -> T:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")

        if mode:
            self.image_encoder.eval()
            self.classifier.train()
        else:
            self.image_encoder.eval()
            self.classifier.eval()

        return self

    def forward(self, batch, device=None):
        device = batch["images"].device if device is None else device

        # get image features and predict
        image_feature = self.encode_image(batch["images"].to(device))
        cls_pred = self.classifier(image_feature)

        out = {"cls_pred": cls_pred, "target_class": batch["labels"].to(device)}
        return out
