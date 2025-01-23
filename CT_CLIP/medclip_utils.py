
import os
import torch
from torch import nn
import torchvision
from transformers import AutoModel
import zipfile

class MedCLIPVisionModelResNet(nn.Module):
    '''
    take resnet50 as backbone.
    '''
    def __init__(self, checkpoint=None, medclip_checkpoint=None):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        num_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fts, 512, bias=False) # projection head
        self.WEIGHTS_NAME = 'pytorch_model.bin'
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, self.WEIGHTS_NAME))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)
        
        # follow like cxr_clip
        del self.model.fc

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, self.WEIGHTS_NAME))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
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
    def __init__(self, checkpoint=None, medclip_checkpoint=None) -> None:
        '''args:
        checkpoint: load from the vision encoder checkpoint
        medclip_checkpoint: load from the vision-text dual encoders checkpoint
        '''
        super().__init__()
        self.vit_type = 'microsoft/swin-tiny-patch4-window7-224' # constants.VIT_TYPE
        self.model = AutoModel.from_pretrained(self.vit_type)
        # self.projection_head = nn.Linear(768, 512, bias=False) #NOTE: no projection head.
        self.WEIGHTS_NAME = 'pytorch_model.bin'
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, self.WEIGHTS_NAME))
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, self.WEIGHTS_NAME))
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

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
        checkpoint_dir=None,
        vision_checkpoint=None
        ) -> None:
        super().__init__()
        assert vision_cls in [MedCLIPVisionModelResNet, MedCLIPVisionModelViT], 'vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]'

        self.vision_model = vision_cls(checkpoint=vision_checkpoint)

        self.WEIGHTS_NAME = 'pytorch_model.bin'
        if checkpoint_dir is not None:
            zipf = zipfile.ZipFile(checkpoint_dir)
            zipf.extractall(checkpoint_dir)
            zipf.close()
            state_dict = torch.load(os.path.join(checkpoint_dir, self.WEIGHTS_NAME))
            self.load_state_dict(state_dict)
            print('load model weight from:', checkpoint_dir)

    # def from_pretrained(self, input_dir=None):
    #     '''
    #     If input_dir is None, download pretrained weight from google cloud and load.
    #     input_dir should be the directory of the zipped file
    #     '''
    #     # unzip
    #     zipf = zipfile.ZipFile(input_dir)
    #     zipf.extractall(input_dir)
    #     zipf.close()
    #     state_dict = torch.load(os.path.join(input_dir, self.WEIGHTS_NAME))
    #     self.load_state_dict(state_dict)
    #     print('load model weight from:', input_dir)
