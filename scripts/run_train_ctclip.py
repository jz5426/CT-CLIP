import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from CTCLIPTrainer import CTClipTrainer
import random
import numpy as np

# ENTRY POINT
def convert_dictconfig_to_dict(cfg):
    if isinstance(cfg, DictConfig):
        return {k: convert_dictconfig_to_dict(v) for k, v in cfg.items()}
    else:
        return cfg

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

    print("Batch Size:", cfg_dot.training_params.batch_size)
    print("Number of Workers:", cfg_dot.training_params.num_workers)
    print("Batch Style:", cfg_dot.training_params.batch_style)
    print("Epoch-Based Patience:", cfg_dot.training_params.epoch_based_patience)
    print("Iteration Evaluate Frequency:", cfg_dot.training_params.iteration_evaluate_frequency)
    print("Learning Rate:", cfg_dot.training_params.learning_rate)
    print("Weight Decay:", cfg_dot.training_params.weight_decay)
    print("Epochs:", cfg_dot.training_params.epochs)

    # convert the config file to dictionary
    cfg = convert_dictconfig_to_dict(cfg_dot)

    tokenizer = BertTokenizer.from_pretrained(
        '/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/CT_CLIP/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        do_lower_case=True,
        local_files_only=True
    )
    text_encoder = BertModel.from_pretrained(
        '/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/CT_CLIP/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
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

    clip = CTCLIP(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_text = 768,
        dim_image = 294912,
        dim_latent = 512,
        downsample_image_embeds = False,
        use_all_token_embeds = False
    )

    # load the pretrained weights
    clip.load('/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/CT-CLIP_v2.pt')

    # uhn cluster
    trainer = CTClipTrainer(
        clip,
        min_epochs=cfg_dot.training_params.min_epochs,
        cfg=cfg,
        tokenizer=tokenizer,
        meta_data='/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/metadata/train_metadata.csv',
        data_train= '/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16',
        data_valid = '/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_val', #TODO:
        reports_file_train = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/radiology_text_reports/train_reports.csv',
        reports_file_valid = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/radiology_text_reports/train_reports.csv', #TODO:
        # reports_file_valid = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/radiology_text_reports/valid_reports.csv',
        labels = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv', #TODO:
        results_folder=f'/cluster/projects/mcintoshgroup/CT-CLIP-CHECKPOINTS', # put the check point in a subdirectory under CT-RATE-CHECKPOINTS 
        # batch_style=cfg_dot.training_params.batch_style,
        batch_size = cfg_dot.training_params.batch_size,
        num_workers = cfg_dot.training_params.num_workers, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
        epoch_based_patience = cfg_dot.training_params.epoch_based_patience,
        iteration_evaluate_frequency = cfg_dot.training_params.iteration_evaluate_frequency,
        wd = cfg_dot.training_params.weight_decay,
        lr = cfg_dot.training_params.learning_rate,
    )
    trainer.train_by_epoch(cfg_dot.training_params.epochs)

if __name__ == '__main__':
    main()