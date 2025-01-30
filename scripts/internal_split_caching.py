"""
cache the internal split for each model baseline, for each dataset (external or internal), mainly used for linear probe evaluation
"""

import torch

from data import CTReportDataSplitter, CTReportXRayClassificationDataset
import os
from cxr_clip_utils import convert_dictconfig_to_dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import random
import numpy as np
from eval_utils import proportion_mapping

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

    # assert cfg_dot.internal_split_caching_params.baseline_type in ['cxr_clip_resnet', 'cxr_clip_swin', 'medclip_resnet', 'medclip_vit', 'gloria_densenet', 'gloria_resnet']
    if 'cxr_clip' in cfg_dot.internal_split_caching_params.baseline_type: # can be either cxr_clip_swin or cxr_clip_resnet
        xray_model_type = cfg_dot.internal_split_caching_params.baseline_type #'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
        pth_base_name = 'swin_cxr_xray_features.pth' if 'swin' in xray_model_type else 'resnet_cxr_xray_features.pth'
        saving_base_name = 'swin_cxr_xray_datasplit.pth' if 'swin' in xray_model_type else 'resnet_cxr_xray_datasplit.pth'
    elif cfg_dot.internal_split_caching_params.baseline_type == 'medclip_resnet':
        xray_model_type = cfg_dot.internal_split_caching_params.baseline_type
        pth_base_name = 'resnet_medclip_features.pth'
        saving_base_name = 'resnet_medclip_datasplit.pth'
        # place this somewhere in the medclip code to remove the learnt fc connected layer at the end, just like cxr_clip: del self.resnet.fc
    elif cfg_dot.internal_split_caching_params.baseline_type == 'medclip_vit':
        xray_model_type = cfg_dot.internal_split_caching_params.baseline_type
        pth_base_name = 'swin_medclip_features.pth'
        saving_base_name = 'swin_medclip_datasplit.pth'
    elif cfg_dot.internal_split_caching_params.baseline_type == 'gloria_densenet':
        xray_model_type = cfg_dot.internal_split_caching_params.baseline_type
        pth_base_name = 'densenet_gloria_features.pth'
        saving_base_name = 'densenet_gloria_datasplit.pth'
    elif cfg_dot.internal_split_caching_params.baseline_type == 'gloria_resnet':
        xray_model_type = cfg_dot.internal_split_caching_params.baseline_type
        pth_base_name = 'resnet_gloria_features.pth'
        saving_base_name = 'resnet_gloria_datasplit.pth'
    else:
        xray_model_type = cfg_dot.internal_split_caching_params.baseline_type
        pth_base_name = f'{xray_model_type}_xray_features.pth'
        saving_base_name = f'{xray_model_type}_datasplit.pth'

    if cfg_dot.internal_split_caching_params.evaluation_dataset == 'ct_rate':

        # base on the baseline model, load the corresponding xray features
        xray_feature_path = f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/xray_features_embeddings/train/{pth_base_name}'
        train_xray_features = torch.load(xray_feature_path)

        # Set up the dataset and data loaders
        #NOTE: the label is the mimic version (with 11 labels) but the report and the data are the original CT-RATE
        train_data_splitter = CTReportDataSplitter(
            csv_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
            labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_mimic_labels.csv', #NOTE: the label need to be the mimic version
            data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/train_preprocessed_xray_mha',
        )
        train_sample, internal_val_samples = train_data_splitter.prepare_samples(
            train_split=cfg_dot.internal_split_caching_params.train_data_portion,
            val_split=0.2
        ) # validation split is always, train_split is controlable

        train_dataset = CTReportXRayClassificationDataset(
            cfg=cfg,
            data=train_sample, # actual data potentially with the embeddings
            data_embeddings=train_xray_features,
            split='train'
        )

        internal_val_dataset = CTReportXRayClassificationDataset(
            cfg=cfg,
            data=internal_val_samples, # actual data potentially with the embeddings
            data_embeddings=train_xray_features,
            split='train'
        )

        # save as a dictionary
        results = {
            'dataset': cfg_dot.internal_split_caching_params.evaluation_dataset,
            'model': cfg_dot.internal_split_caching_params.baseline_type,
            'proportion': cfg_dot.internal_split_caching_params.train_data_portion,
            'train_split': train_dataset,
            'internal_val_split': internal_val_dataset
        }
        
        # NOTE: save the object so that it is differentateid based on dataset, proportion, and baseline.
        file_path = f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/lp_internal_splits/{proportion_mapping(cfg_dot.internal_split_caching_params.train_data_portion)}/{saving_base_name}'
        os.makedirs(file_path, exist_ok=True)
        torch.save(results, file_path)

    elif cfg_dot.internal_split_caching_params.evaluation_dataset == 'vinBig': # the ct dataset
        print('not implemented yet')
        pass

# Example usage
if __name__ == "__main__":
    main()