
import os
import torch
from torch import nn
import torchvision
from transformers import AutoModel

class MedCLIPVisionModelResNet(nn.Module):
    '''
    take resnet50 as backbone.
    '''
    def __init__(self, medclip_checkpoint=None):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=False) # prevent from download everything
        num_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fts, 512, bias=False) # projection head
        self.WEIGHTS_NAME = 'pytorch_model.bin'
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)
        else:
            print('NOT LOADING ANY MEDICAL RELATED PRETRAINED WEIGHTS')
        
    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, self.WEIGHTS_NAME))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)

        # find the intersection
        model_keys = set(self.state_dict().keys()) # this model's own dictionary
        ckpt_keys = set(new_state_dict.keys()) # the pretrained dictionary
        loaded_keys = ckpt_keys.intersection(model_keys) - set(missing_keys)
        assert (len(self.model.state_dict().keys())) == len(loaded_keys) # check the model is indeed successfully loaded.

        # print('missing keys:', missing_keys)
        # print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

    def forward(self, pixel_values, **kwargs):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        img_embeds = self.model(pixel_values)
        return img_embeds

class MedCLIPVisionModelViT(nn.Module):
    '''take an VIT model as the backbone.
    '''
    def __init__(self, medclip_checkpoint=None) -> None:
        '''args:
        checkpoint: load from the vision encoder checkpoint
        medclip_checkpoint: load from the vision-text dual encoders checkpoint
        '''
        super().__init__()
        self.WEIGHTS_NAME = 'pytorch_model.bin'
        self.vit_type = 'microsoft/swin-tiny-patch4-window7-224' # constants.VIT_TYPE
        self.cache_dir = '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/cxr-swinTiny/'
        self.model = AutoModel.from_pretrained(
            self.vit_type, add_pooling_layer=False, cache_dir=self.cache_dir, local_files_only=True
        )
        self.projection_head = nn.Linear(768, 512, bias=False)

        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)
        else:
            print('NOT LOADING ANY MEDICAL RELATED PRETRAINED WEIGHTS')

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, self.WEIGHTS_NAME))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, _ = self.load_state_dict(new_state_dict, strict=False)

        # find the intersection
        model_keys = set(self.state_dict().keys()) # this model's own dictionary
        ckpt_keys = set(new_state_dict.keys()) # the pretrained dictionary
        loaded_keys = ckpt_keys.intersection(model_keys) - set(missing_keys)
        assert (len(self.model.state_dict().keys()) + 1) == len(loaded_keys) # check the model is indeed successfully loaded.

        # print('missing keys:', missing_keys)
        # print('unexpected keys:', unexpected_keys)
        print('medclip ViT loads model weight from:', checkpoint)

    def forward(self, pixel_values, project=True):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        output = self.model(pixel_values)
        img_embeds = output['pooler_output']
        # if project:
        #     img_embeds = self.projection_head(img_embeds)
        return img_embeds


class MedCLIPVisionModel(nn.Module):
    def __init__(self,
        vision_cls=MedCLIPVisionModelResNet,
        checkpoint=None,
        ) -> None:
        super().__init__()
        assert vision_cls in [MedCLIPVisionModelResNet, MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'
        self.vision_model = vision_cls(medclip_checkpoint=checkpoint)