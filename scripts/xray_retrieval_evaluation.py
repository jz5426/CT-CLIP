"""
run the following script to check the retrieval performance using xray as query.

TODO:
repeat the following and accumulate the stats:
    1. forward pass a synethic xray for each CT image in the validation set, to the pretrained xray encoder from ULIP style training
    2. with the embedding from xray encoder, compare it with the existing embeddings of the CT images based the image_embeddings.pth and find the top-k accuracy
        - note that since we forward pass the syntheic xray of examing ct image, we know the ground truth based on the dictionary id from image_embeddings.
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
from CTCLIPTrainer import CTClipTrainer



@hydra.main(
        version_base=None,
        # config_path="C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\configs",
        # config_path="/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/configs",
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

    # windows wsl from download    
    # tokenizer = BertTokenizer.from_pretrained(
    #     'microsoft/BiomedVLP-CXR-BERT-specialized',
    #     do_lower_case=True,
    #     cache_dir='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/predownloaded_models/BertTokenizer')
    # text_encoder = BertModel.from_pretrained(
    #     'microsoft/BiomedVLP-CXR-BERT-specialized',
    #     cache_dir='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/predownloaded_models/BertModel'
    #     )

    # windows wsl from local files
    # tokenizer = BertTokenizer.from_pretrained(
    #     '/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/predownloaded_models/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
    #     do_lower_case=True,
    #     local_files_only=True)
    # text_encoder = BertModel.from_pretrained(
    #     '/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
    #     local_files_only=True
    #     )

    # uhn cluster from local filesc
    #TODO: 
        # 1. copy the downloaded huggingface model in G:\Chris\CT-CLIP\predownloaded_models (shield external drive) to the CT-CLIP
        # 2. for the image_encoder section of the yaml file (such as clip_Swin_clincial), replace the directory to the correct one
    tokenizer = BertTokenizer.from_pretrained(
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        do_lower_case=True,
        local_files_only=True
    )
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
    # generic command to load the pretrained xray encoder weights and freeze the parameters.
    clip_xray.load_xray_encoder('path_to_pretrained_xray_encoder_weights_{}'.format(ckpt_name), freeze_weights=True)

    # check the trainable parameters
    xray_encoder_trainable = sum(p.numel() for p in clip_xray.xray_encoder.parameters() if p.requires_grad)
    ct_clip_trainable = sum(p.numel() for p in clip_xray.CTCLIP.parameters() if p.requires_grad)
    assert(xray_encoder_trainable > 0)
    assert(ct_clip_trainable == 0)

    # from run_train.py
    retrival_evaluator = CTClipTrainer(
        clip_xray,
        cfg=cfg,
        tokenizer=tokenizer,
        batch_style='experiment',
        data_train= '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/train_preprocessed_xray_mha',
        data_valid = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/valid_preprocessed_xray_mha',
        img_embedding_paths = {
            'train': '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/train/image_features.pth', 
            'valid': '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/valid/image_features.pth'
        },
        text_embedding_paths = {
            'train': '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/train/text_features.pth',
            'valid': '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/valid/text_features.pth'
        },
        reports_file_train = '/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
        reports_file_valid = '/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/valid_reports.csv',
        labels = '/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        batch_size = 360,
        results_folder='/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS',
        num_train_steps = -1, # placeholder
        num_workers = 10, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
        train_from_scratch = False
    )  
    retrival_evaluator.extract_xray_features('/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/features_embeddings_correct')
    
    # retrival evaluation on the ct modality, queried by the xray
    # retrival_evaluator.retrieval_evaluation(latent_type='ct', split='valid', topk=[1, 5, 10, 50])

    # retrival evaluation on the report modality, qurired by the xray
    # retrival_evaluator.retrieval_evaluation(latent_type='report', split='valid', topk=[1, 5, 10, 50])

if __name__ == '__main__':
    main()