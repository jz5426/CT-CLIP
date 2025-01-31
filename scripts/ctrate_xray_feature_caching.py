"""
this script responsible for caching the xray features from the assigned baseline.
"""

import torch
from transformers import BertTokenizer, BertModel
import os
from cxr_clip_utils import convert_dictconfig_to_dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformer_maskgit import CTViT
from transformers import BertModel
from ct_clip import CTCLIPwithXray
import random
import numpy as np
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

    # seed everything
    seed = 1024
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    run(cfg)


def run(cfg_dot):

    # convert the config file to dictionary
    cfg = convert_dictconfig_to_dict(cfg_dot)

    torch.cuda.empty_cache()
    text_encoder = BertModel.from_pretrained(
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        local_files_only=True
    )
    tokenizer = BertTokenizer.from_pretrained(
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        do_lower_case=True,
        local_files_only=True)

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

    # assert cfg_dot.xray_feature_caching_params.baseline_type in ['cxr_clip_resnet', 'cxr_clip_swin', 'medclip_resnet', 'medclip_vit', 'gloria_densenet', 'gloria_resnet']
    if 'cxr_clip' in cfg_dot.xray_feature_caching_params.baseline_type: # can be either cxr_clip_swin or cxr_clip_resnet
        xray_model_type = cfg_dot.xray_feature_caching_params.baseline_type #'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
        dim_xray = 768 if 'swin' in cfg_dot.xray_feature_caching_params.baseline_type else 2048  # if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048
        pth_base_name = 'swin_cxr_xray_features.pth' if 'swin' in xray_model_type else 'resnet_cxr_xray_features.pth'
        latent_size = 512
    elif cfg_dot.xray_feature_caching_params.baseline_type == 'medclip_resnet':
        xray_model_type = cfg_dot.xray_feature_caching_params.baseline_type
        dim_xray = 2048
        pth_base_name = 'resnet_medclip_features.pth'
        latent_size = 512
        # place this somewhere in the medclip code to remove the learnt fc connected layer at the end, just like cxr_clip: del self.resnet.fc
    elif cfg_dot.xray_feature_caching_params.baseline_type == 'medclip_vit':
        xray_model_type = cfg_dot.xray_feature_caching_params.baseline_type
        dim_xray = 768
        pth_base_name = 'swin_medclip_features.pth'
        latent_size = 512
    elif cfg_dot.xray_feature_caching_params.baseline_type == 'gloria_densenet':
        xray_model_type = cfg_dot.xray_feature_caching_params.baseline_type
        dim_xray = 1024 #TODO: double check this.
        pth_base_name = 'densenet_gloria_features.pth'
        latent_size = 768
    elif cfg_dot.xray_feature_caching_params.baseline_type == 'gloria_resnet':
        xray_model_type = cfg_dot.xray_feature_caching_params.baseline_type
        dim_xray = 2048
        pth_base_name = 'resnet_gloria_features.pth'
        latent_size = 768 # the final size of the xray embedding is indeed different in gloria
    else:
        xray_model_type = cfg_dot.xray_feature_caching_params.baseline_type
        dim_xray = 768 if 'swin' in cfg_dot.xray_feature_caching_params.baseline_type.lower() else 2048
        pth_base_name = f'{xray_model_type}_xray_features.pth'
        latent_size = 512

    clip_xray = CTCLIPwithXray(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_text = 768,
        dim_image = 294912,
        xray_model_type = xray_model_type,
        dim_xray = dim_xray, # output size of the xray feature extractor
        dim_latent = latent_size, # latent size that match the CT vision encoder and the text encoder.
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False,
        cfg=cfg,
        auto_load_pretrained_weights=True # NOTE: automatically load the model weights based on the xray_model_type
    )

    ###################################### NOTE: the following can be cached ######################################

    split = 'train'
    train_split_inference = ct_rate_split(split, xray_model_type, clip_xray, cfg, cfg_dot, tokenizer)
    # get xray latent features from this particularly baseline model
    train_split_inference.xray_feature_extraction(
        directory=f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/xray_features_embeddings/',
        pth_name=pth_base_name, 
        append=True
    )

    split = 'valid'
    valid_split_inference = ct_rate_split(split, xray_model_type, clip_xray, cfg, cfg_dot, tokenizer)
    valid_split_inference.xray_feature_extraction(
        directory=f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/xray_features_embeddings/',
        pth_name=pth_base_name, 
        append=True
    )

    ###################################### NOTE: the above can be cached ######################################

    print(f'Finished caching the xray feature extracted from the baseline: {cfg_dot.xray_feature_caching_params.baseline_type}')

def ct_rate_split(split, xray_model_type, clip_xray, cfg, cfg_dot, tokenizer):

    split_inference = CTClipInference(
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
        batch_size = cfg_dot.xray_feature_caching_params.batch_size,
        num_train_steps = -1, # placeholder
        num_workers = cfg_dot.xray_feature_caching_params.num_workers, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
        feature_extraction_mode = True # might be optional
    )  

    return split_inference

# Example usage
if __name__ == "__main__":
    main()