import torch
import torch.nn as nn

from .. import builder
from .. import loss
from transformers import AutoTokenizer

def build_text_model(cfg):
    return models.text_model.BertEncoder(cfg)

class GLoRIA(nn.Module):
    def __init__(self, cfg):
        super(GLoRIA, self).__init__()

        self.cfg = cfg
        self.text_encoder = builder.build_text_model(cfg)
        self.img_encoder = builder.build_img_model(cfg)

        self.local_loss = loss.gloria_loss.local_loss
        self.global_loss = loss.gloria_loss.global_loss
        self.local_loss_weight = self.cfg.model.gloria.local_loss_weight
        self.global_loss_weight = self.cfg.model.gloria.global_loss_weight

        self.temp1 = self.cfg.model.gloria.temp1
        self.temp2 = self.cfg.model.gloria.temp2
        self.temp3 = self.cfg.model.gloria.temp3
        self.batch_size = self.cfg.train.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs):
        img_feat_g, img_emb_l = self.img_encoder(imgs, get_local=True)
        img_emb_g, img_emb_l = self.img_encoder.generate_embeddings(
            img_feat_g, img_emb_l
        )

        return img_emb_l, img_emb_g

    def forward(self, x):

        # img encoder branch
        img_emb_l, img_emb_g = self.image_encoder_forward(x["imgs"])

        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )

        return img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents
