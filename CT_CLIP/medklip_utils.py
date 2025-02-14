# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
import copy
from typing import Optional
import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F
# from .transformer import *
import torchvision.models as models
from einops import rearrange
from transformers import AutoModel

class MedKLIP(nn.Module):

    def __init__(self):
        super(MedKLIP, self).__init__()

        self.d_model = 256

        ''' visual backbone'''
        self.resnet_dict = {"resnet50": models.resnet50(pretrained=False)}
        self.resnet = self._get_res_basemodel('resnet50')
        num_ftrs = int(self.resnet.fc.in_features/2)
        self.res_features = nn.Sequential(*list(self.resnet.children())[:-3])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, self.d_model)

        self.apply(self._init_weights)

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def image_encoder(self, xis):
        #patch features
        """
        16 torch.Size([16, 1024, 14, 14])
        torch.Size([16, 196, 1024])
        torch.Size([3136, 1024])
        torch.Size([16, 196, 256])
        """
        batch_size = xis.shape[0]
        res_fea = self.res_features(xis) #batch_size,feature_size,patch_num,patch_num
        res_fea = rearrange(res_fea,'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(res_fea,'b n d -> (b n) d')
        #batch_size,num,feature_size
        # h = h.squeeze()
        x = self.res_l1(h)
        x = F.relu(x)
        x = self.res_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        return out_emb

    def forward(self, images):

        # labels batch,51,75 binary_label batch,75 sample_index batch,index
        B = images.shape[0]
        ''' Visual Backbone '''
        x = self.image_encoder(images) #batch_size,patch_num,dim
        x = x.mean(dim=1)
        return x
        # x = self.resnet.avgpool(x)
        # x = torch.flatten(x, 1)
        # features = x.transpose(0,1) #patch_num b dim

        # #query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
        # query_embed = self.disease_embedding_layer(self.disease_book)
        # query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
        # features,ws = self.decoder(query_embed, features, 
        #     memory_key_padding_mask=None, pos=None, query_pos=None)
        # out = self.dropout_feas(features)
        # anatomy_query = self.ana_book[smaple_index,:] # batch, Q , position_num ,dim
        # # [Q,B,A]
        # ll = out.transpose(0,1) # B Q A
        # Q = ll.shape[1]
        # ll = ll.reshape(ll.shape[0]*ll.shape[1],-1)
        # ll = self.cl_fc(ll)
        # ll = ll.unsqueeze(dim =-1)
        # return features

    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()