import os
from cxr_clip_utils import convert_dictconfig_to_dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
import random
import numpy as np
from retrieval_evaluation_utils import ctrate_retrieval_evaluation, mimic_retrieval_evaluation

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
    torch.cuda.empty_cache()

    cfg = convert_dictconfig_to_dict(cfg_dot)
    # windows wsl from local files
    tokenizer = BertTokenizer.from_pretrained(
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        do_lower_case=True,
        local_files_only=True)
    text_encoder = BertModel.from_pretrained(
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        local_files_only=True
        )

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

    print('Starting Xray related retrieval experiments')
    # our retrival results: from cxr_clip model, from our pretrained xray encoder distilled from ct_clip NOTE: shared
    baselines = [
        ## newly add
        'modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch',
        'modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch'
        ## baseline pretrained model (not pretrained by us)
        # 'cxr_clip_swin', # xray encoder weights from cxr_clip
        # 'cxr_clip_resnet',
        # 'medclip_resnet',
        # 'medclip_vit',
        ## our pretrained model
        # 'modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch',
        # 'modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch',
        ## missing
        # 'gloria_densenet',
        # 'gloria_resnet',
    ]

    params = {
        'cfg': cfg,
        'baselines': baselines,
        'image_encoder': image_encoder,
        'text_encoder': text_encoder,
        'tokenizer': tokenizer,
        'metric_results_destination': ''
    }
    if cfg_dot.retrieval_params.evaluation_dataset == 'ct-rate':
        params['metric_results_destination'] = './ct-rate_retrieval_results'
        ctrate_retrieval_evaluation(params)
    elif cfg_dot.retrieval_params.evaluation_dataset == 'mimic':
        params['metric_results_destination'] = './mimic_retrieval_results'
        mimic_retrieval_evaluation(params)

if __name__ == '__main__':

    main()