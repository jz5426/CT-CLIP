"""
- this script is intended to merge the external_linear_probing and internal_linear_probing evaluation

note that this file depends on the following are done:
- the xray feature for each baseline is cached using the ctrate_xray_feature_caching.py
- the internal split is already cached using the internal_split_caching.py
- the implementation of the linear_probe_utils.py, which depends on above.
"""

import torch

from linear_probe_utils_mnt import evaluate_classifier, get_train_internal_split, get_pathologies, linear_probing_main
from eval_utils import LinearProbeModel, metadata_base_on_model_type, save_metric_results
from transformers import BertModel
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
import pandas as pd
import constants as const
from collections import defaultdict

@hydra.main(
        version_base=None,
        config_path="/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/chris/CT-CLIP/configs",
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

    # iterate 10 times and collect the stats
    if cfg.linear_probing_params.multi_sweep_evaluation:
        # List of seeds to iterate over
        seed_list = [1024, 1234, 4321, 5678, 8765, 1357, 2468, 9753, 8642, 3141]
        seed_dicts = []
        for i, seed in enumerate(seed_list):
            # Set the random seeds
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Run the function and store the label_predictions
            metric_results = run(cfg)
            seed_dicts.append(metric_results)
            print(f'finish the evaluation round {i+1}/{len(seed_list)}')

        # merge all of the seed dictionaries
        merged_dict = defaultdict(list)
        merged_dict[const.SEED] = seed_list
        for d in seed_dicts:
            for key, value in d.items():
                merged_dict[key].extend(value)        

        # save the results
        save_metric_results(
            const.EXPERIMENT_RESULTS_SAVING_PATH,
            'linear_probe_multiRun_results.csv',
            pd.DataFrame(dict(merged_dict)),
            cfg.linear_probing_params.override_metric_results)

    # seed everything
    seed = 1024
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    # NOTE: external few shot
    # portions = [0.01, 0.025, 0.05, 0.1, 1.]
    # evaluation_datasets = ['radchest_ct_pure', 'mimic', 'ct-rate']

    # NOTE: internal few shot
    portions = [0.1, 0.2, 0.5]
    evaluation_datasets = ['radchest_all_disease_ct_only_internal', 'ct-rate']
    for p in portions:
        for eval_data in evaluation_datasets:
            cfg.linear_probing_params.train_data_portion = p
            cfg.linear_probing_params.evaluation_dataset = eval_data
            metric_results = run(cfg)
            # NOTE: everything is saved to the same file.
            # save it to a csv file
            metric_results[const.SEED] = [seed]
            save_metric_results(
                const.EXPERIMENT_RESULTS_SAVING_PATH,
                f'{cfg.linear_probing_params.baseline_type}_linear_probe_results.csv',
                pd.DataFrame(metric_results),
                cfg.linear_probing_params.override_metric_results)


def run(cfg_dot):
    # NOTE: this script only works for bi-mamba
    assert(cfg_dot.linear_probing_params.baseline_type == 'bi-mamba')

    # convert the config file to dictionary
    cfg = convert_dictconfig_to_dict(cfg_dot)

    torch.cuda.empty_cache()
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

    dim_xray, xray_model_type, pth_base_name, latent_size = metadata_base_on_model_type(
        cfg_dot.linear_probing_params.baseline_type,
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
    # TODO: toggle the path here.
    # train_dataset, internal_val_dataset = get_train_internal_split(cfg_dot, cfg)
    datasets = get_train_internal_split(cfg_dot, cfg)
    train_dataset = datasets['train_dataset']
    internal_val_dataset = datasets['internal_val_dataset']
    test_dataset = None
    if 'test_dataset' in datasets:
        test_dataset = datasets['test_dataset']
    
    pathologies = get_pathologies(dataset=cfg_dot.linear_probing_params.evaluation_dataset)
    
    # NOTE: perform linear probing training

    # Initialize the wrapper model for either NOTE: linear probe or full model finetuninng
    # that is, add a additional fc layer on top of the vision model and the feature_projector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearProbeModel(in_features=latent_size, num_classes=len(pathologies))
    model.to(device)

    # pth_base_name = f'{pth_base_name}__train_portion_{cfg_dot.linear_probing_params.train_data_portion}'
    classifier_ckpt_base_name = f'{pth_base_name}__train_portion_{cfg_dot.linear_probing_params.train_data_portion}'

    parent_dir = cfg_dot.linear_probing_params.evaluation_dataset
    ckpt_parent_dir = os.path.join(cfg_dot.linear_probing_params.mnt_cpt_dest, parent_dir)
    best_ckpt_destination = os.path.join(ckpt_parent_dir, f'{classifier_ckpt_base_name}_best_model.pth')
    params = {
        'num_classes': len(pathologies),
        'latent_size': latent_size,
        'train_dataset': train_dataset,
        'internal_val_dataset': internal_val_dataset,
        'cfg_dot': cfg_dot,
        'ckpt_parent_dir': ckpt_parent_dir,
        'best_ckpt_destination': best_ckpt_destination,
        'model': model,
        'device': device
    }
    model = linear_probing_main(params)

    # NOTE: pay attention that mimic and ct-rate dataset load different model checkpoints
    #   mimic load the classifer only but with a additional backbone
    #   ct-rate only has the classifier
    #   vinBig TBD
    params = {
        'dataset': cfg_dot.linear_probing_params.evaluation_dataset,
        'cfg': cfg,
        'cfg_dot': cfg_dot,
        'clip_xray': clip_xray,
        'device': device,
        'xray_model_type': xray_model_type,
        'model': model, # the linear classifier
        'best_ckpt_destination': best_ckpt_destination,
        'classifier_ckpt_base_name': classifier_ckpt_base_name,
        'pth_base_name': pth_base_name, # mainly for the ct-rate dataset
        'test_data': test_dataset
    }
    # TODO: toggle the path here
    metric_results = evaluate_classifier(params)

    return metric_results

# Example usage
if __name__ == "__main__":
    main()