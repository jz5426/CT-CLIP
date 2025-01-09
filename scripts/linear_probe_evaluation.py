"""
TODO:
- train a classifier layer on top of the pretrained xray model (either our pretrained version or the one from cxr-clip)
    - based on the multi-labels from the training data (evaluate based on the validation data)
- compare the performance
"""

import os
from cxr_clip_utils import convert_dictconfig_to_dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIPwithXray
import random
import numpy as np
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from zero_shot import CTClipInference

@hydra.main(
        version_base=None,
        config_path="/cluster/home/t135419uhn/CT-CLIP/configs",
        config_name="train")
def main(cfg: DictConfig):

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

    # seed_everything(1234)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # efficient performance optimization.

    cfg = convert_dictconfig_to_dict(cfg)

    # seed everything
    seed = 1024
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    run(cfg)


def run(cfg):
    torch.cuda.empty_cache()
    #NOTE: you need to use the follownig command to copy and past to the location cp -rL /path/to/source_directory /path/to/destination_directory 
        # the copied files in the destination folder will behave like regular files and directories. You can copy and paste them as usual using a file manager

    # uhn cluster from local filesc
    tokenizer = BertTokenizer.from_pretrained(
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        do_lower_case=True,
        local_files_only=True)
    text_encoder = BertModel.from_pretrained(
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        local_files_only=True
        )

    print("---------")
    print(tokenizer.pad_token_id)
    print(tokenizer.mask_token_id)
    print("-----------")


    image_encoder = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 480,
        patch_size = 20,
        temporal_patch_size = 10,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    )
    #dim_image = 131072,

    clip_xray = CTCLIPwithXray(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        # tokenizer=tokenizer,
        dim_text = 768,
        dim_image = 294912,
        xray_model_type = 'swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'resnet',
        dim_xray = 768 if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False,
        cfg=cfg
    )

    # NOTE: load the pretrained backbones
    ckpt_name = 'r50_mcc.tar' if cfg['model']['image_encoder']['name'] == 'resnet' else 'swint_mcc.tar'

    # NOTE: cxr-clip pretrained weights
    # clip_xray.load_xray_encoder(
    #     '/cluster/home/t135419uhn/CT-CLIP/models/cxr_clip/{}'.format(ckpt_name), # cxr-clip pretrained
    #     freeze_weights=True
    # )
    # pth_name = 'cxr_xray_features.pth'

    # NOTE: for our pretrained model: load the whole thing
    ckp_name = 'CTClip.lowest_val_cl_loss_during_iterations'
    clip_xray.load_pretrained_ct_xray_clip(f'/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/{ckp_name}.pt')
    pth_name = f'{ckp_name}_xray_features.pth'

    # check the trainable parameters
    xray_encoder_trainable = sum(p.numel() for p in clip_xray.xray_encoder.parameters() if p.requires_grad)
    ct_clip_trainable = sum(p.numel() for p in clip_xray.CTCLIP.parameters() if p.requires_grad)
    # assert(xray_encoder_trainable == 0)
    # assert(ct_clip_trainable == 0)

    split = 'valid'
    retrival_evaluator = CTClipInference(
        clip_xray,
        cfg=cfg,
        tokenizer=tokenizer,
        data_folder= f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/{split}_preprocessed_xray_mha',
        # NOTE: the embedding paths are MANDATORY for the dataloader to work. RUN THIS SCRIPT MAINLY AFTER THE CTCLIP EMBEDDINGS ARE EXTRACTED.
        img_embedding_paths = {
            f'{split}': f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/{split}/image_features.pth'
        },
        text_embedding_paths = {
            f'{split}': f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/{split}/text_features.pth'
        },
        reports_file = f'/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/{split}_reports.csv',
        labels = f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
        results_folder="./inference_zeroshot_retrieval",
        batch_size = 128,
        num_train_steps = -1, # placeholder
        num_workers = 10, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
        feature_extraction_mode = True # might be optional
    )  

    embedding_directory = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/'
    xray_features = retrival_evaluator.xray_feature_extraction(embedding_directory, pth_name=pth_name)

    # TODO: create a dataset with the xray features and the labels

    # TODO: with the xray features, feed that into a one layer classifier for training.


if __name__ == '__main__':
    main()