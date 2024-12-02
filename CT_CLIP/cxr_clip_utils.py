from typing import Dict, Union
from omegaconf import DictConfig, OmegaConf
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
import hydra
import os

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
                self.image_encoder = SwinModel.from_pretrained(name)
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
                # TODO: add vision models if needed
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
            # TODO: add vision models if needed
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


@hydra.main(version_base=None, config_path="configs", config_name="train")
def get_config(cfg: DictConfig):

    OmegaConf.resolve(cfg)

    if "LOCAL_RANK" in os.environ:
        # for ddp
        # passed by torchrun or torch.distributed.launch
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # for debugging
        local_rank = -1

    if local_rank < 1:
        print(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(1234)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    cfg = convert_dictconfig_to_dict(cfg)
    return cfg
