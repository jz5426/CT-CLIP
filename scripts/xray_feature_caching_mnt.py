"""
the mnt version of the xray_feature_caching.py file.
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
from eval_utils import metadata_base_on_model_type
from zero_shot import CTClipInference, VinBigDataChestXrayInference
import constants as const

@hydra.main(
        version_base=None,
        config_path="/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/chris/CT-CLIP/configs", #"/cluster/home/t135419uhn/CT-CLIP/configs"
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
    # text_encoder = BertModel.from_pretrained(
    #     '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
    #     local_files_only=True
    # )
    # tokenizer = BertTokenizer.from_pretrained(
    #     '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
    #     do_lower_case=True,
    #     local_files_only=True)
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

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

    # get the necessary metadata
    dim_xray, xray_model_type, pth_base_name, latent_size = metadata_base_on_model_type(
        cfg_dot.xray_feature_caching_params.baseline_type,
        pth_trailing_string='features')

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

    pass

    if cfg_dot.xray_feature_caching_params.evaluation_dataset == 'ct-rate':
        split = 'train'
        train_split_inference = ct_rate_split(split, clip_xray, cfg, cfg_dot, tokenizer)
        # get xray latent features from this particularly baseline model
        train_split_inference.xray_feature_extraction(
            directory='/mnt/g/Chris/CT-RATE-FINAL/processed_dataset/xray_features_embeddings',
            pth_name=pth_base_name, 
            append=True
        )
        # /cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/xray_features_embeddings/
        split = 'valid'
        valid_split_inference = ct_rate_split(split, clip_xray, cfg, cfg_dot, tokenizer)
        valid_split_inference.xray_feature_extraction(
            directory='/mnt/g/Chris/CT-RATE-FINAL/processed_dataset/xray_features_embeddings',
            pth_name=pth_base_name, 
            append=True
        )
        print(f'Finished caching the xray feature of {cfg_dot.xray_feature_caching_params.evaluation_dataset} extracted from the baseline: {cfg_dot.xray_feature_caching_params.baseline_type}')
        return 

    if cfg_dot.xray_feature_caching_params.evaluation_dataset in [const.RADCHEST_CT_PURE_INTERNAL, const.RADCHEST_CT_INTERNAL]:
        radchestct_evaluator = radchest_ct_split(clip_xray, cfg, cfg_dot, tokenizer)
        radchestct_evaluator.xray_feature_extraction(
            directory=f'/mnt/g/radchest_preprocessed/{cfg_dot.linear_probing_params.evaluation_dataset}/features_embeddings',
            pth_name=pth_base_name, 
            append=True
        )
        print(f'Finished caching the xray feature of {cfg_dot.linear_probing_params.evaluation_dataset} extracted from the baseline: {cfg_dot.xray_feature_caching_params.baseline_type}')
        return

def ct_rate_split(split, clip_xray, cfg, cfg_dot, tokenizer):
    data_folder = f'/mnt/g/Chris/CT-RATE-FINAL/processed_dataset/{split}_preprocessed_xray_mha'
    img_embedding_path = f'/mnt/g/Chris/CT-RATE-FINAL/processed_dataset/features_embeddings_correct/{split}/image_features.pth'
    text_embedding_path = f'/mnt/g/Chris/CT-RATE-FINAL/processed_dataset/features_embeddings_correct/{split}/text_features.pth'
    reports_file = f'/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/chris/CT-CLIP/dataset/radiology_text_reports/{split}_reports.csv'
    labels = f'/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv'

    split_inference = CTClipInference(
        clip_xray,
        cfg=cfg,
        tokenizer=tokenizer,
        data_folder=data_folder,
        # NOTE: the embedding paths are MANDATORY for the dataloader to work. RUN THIS SCRIPT MAINLY AFTER THE CTCLIP EMBEDDINGS ARE EXTRACTED.
        img_embedding_paths = {
            f'{split}': img_embedding_path
        },
        text_embedding_paths = {
            f'{split}': text_embedding_path
        },
        reports_file = reports_file,
        labels = labels,
        results_folder="./inference_zeroshot_retrieval",
        batch_size = cfg_dot.xray_feature_caching_params.batch_size,
        num_train_steps = -1, # placeholder
        num_workers = cfg_dot.xray_feature_caching_params.num_workers, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
        feature_extraction_mode = True # might be optional
    )  

    return split_inference


def radchest_ct_split(clip_xray, cfg, cfg_dot, tokenizer):

    if cfg_dot.xray_feature_caching_params.evaluation_dataset == const.RADCHEST_CT_PURE_INTERNAL:
        label_file = 'final_labels_pure_clean.csv'
    elif cfg_dot.xray_feature_caching_params.evaluation_dataset == const.RADCHEST_CT_INTERNAL:
        label_file = 'final_labels_clean.csv'

    split_inference = CTClipInference(
        clip_xray,
        tokenizer=tokenizer,
        cfg=cfg,
        data_folder = '/mnt/g/radchest_preprocessed/preprocessed_xray_mha',
        labels = f'/mnt/g/radchest_preprocessed/{label_file}',
        batch_size = cfg_dot.xray_feature_caching_params.batch_size,
        num_workers = cfg_dot.xray_feature_caching_params.num_workers, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
        results_folder="inference_zeroshot/",
        num_train_steps = 1,
        feature_extraction_mode = True, # extract only the text and ct features only
        dataset=const.RADCHEST_XRAY # this is what differentiate with ct-rate one.
    )

    return split_inference

# Example usage
if __name__ == "__main__":
    main()