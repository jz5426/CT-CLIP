import os
from cxr_clip_utils import convert_dictconfig_to_dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
# from torch_geometric import seed_everything
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer, CTCLIPwithXray
from CTCLIPTrainer import CTClipTrainer
import random
import numpy as np
import argparse

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
    print("Train from Scratch:", cfg_dot.training_params.train_from_scratch)
    print("Epoch-Based Patience:", cfg_dot.training_params.epoch_based_patience)
    print("Iteration Evaluate Frequency:", cfg_dot.training_params.iteration_evaluate_frequency)
    print("Text Contrastive Learning Weight:", cfg_dot.training_params.text_cl_weight)
    print("CT Contrastive Learning Weight:", cfg_dot.training_params.ct_cl_weight)
    print("Learning Rate:", cfg_dot.training_params.learning_rate)
    print("Weight Decay:", cfg_dot.training_params.weight_decay)
    print("Epochs:", cfg_dot.training_params.epochs)
    print("Use Pretrained X-Ray Encoder:", cfg_dot.training_params.use_pretrained_xray_encoder)

    # convert the config file to dictionary
    cfg = convert_dictconfig_to_dict(cfg_dot)

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


    # clip = CTCLIP(
    #     image_encoder = image_encoder,
    #     text_encoder = text_encoder,
    #     dim_text = 768,
    #     dim_image = 294912,
    #     dim_latent = 512,
    #     extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    #     use_mlm=False,
    #     downsample_image_embeds = False,
    #     use_all_token_embeds = False

    # )

    xray_model_type = 'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
    clip_xray = CTCLIPwithXray(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        # tokenizer=tokenizer,
        dim_text = 768,
        dim_image = 294912,
        xray_model_type = xray_model_type,
        dim_xray = 768 if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False,
        cfg=cfg,
        auto_load_pretrained_weights=False # because it loads it later.
    )

    # windows
    # clip_xray.load("C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\Chris\\CT-CLIP\\models\\CT-CLIP_v2.pt",
    #             "C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\Chris\\CT-CLIP\\models\\cxr_clip\\{}".format(ckpt_name))

    # windows wsl
    # clip_xray.load("/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/models/CT-CLIP_v2.pt",
    #             "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/models/cxr_clip/{}".format(ckpt_name))

    # uhn cluster
    #NOTE: if cfg_dot.training_params.use_pretrained_xray_encoder is true => xray encoder and the projection layer is loaded with pretrained cxr_clip weights
    # Load the CT-CLIP pretrained backbone to CT-CLIP and optionlly load the pretrained cxr_clip xray encoder weights
    ckpt_name = 'r50_mcc.tar' if cfg['model']['image_encoder']['name'] == 'resnet' else 'swint_mcc.tar' # NOTE: weights for cxr_clip xray encoder
    clip_xray.load(
        "/cluster/home/t135419uhn/CT-CLIP/models/CT-CLIP_v2.pt",
        f"/cluster/home/t135419uhn/CT-CLIP/models/cxr_clip/{ckpt_name}" if cfg_dot.training_params.use_pretrained_xray_encoder else None
    )

    # check the trainable parameters
    xray_encoder_trainable = sum(p.numel() for p in clip_xray.xray_encoder.parameters() if p.requires_grad)
    ct_clip_trainable = sum(p.numel() for p in clip_xray.CTCLIP.parameters() if p.requires_grad)
    assert(xray_encoder_trainable > 0)
    assert(ct_clip_trainable == 0)

    # windows
    # trainer = CTClipTrainer(
    #     clip_xray,
    #     cfg=cfg,
    #     data_train= "G:\\Chris\\processed_data\\train_preprocessed_xray_mha",
    #     data_valid = "G:\\Chris\\processed_data\\train_preprocessed_xray_mha",
    #     labels = "C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\dataset\\multi_abnormality_labels\\dataset_multi_abnormality_labels_valid_predicted_labels.csv",
    #     batch_size = 2,
    #     results_folder=".\\test",
    #     num_train_steps = 100001,
    #     num_workers = 1,
    # )

    # windows wsl
    # trainer = CTClipTrainer(
    #     clip_xray,
    #     cfg=cfg,
    #     tokenizer=tokenizer,
    #     batch_style='patient',
    #     data_train= "/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/train_preprocessed_xray_mha",
    #     data_valid = "/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/valid_preprocessed_xray_mha",
    #     img_embedding_paths = {
    #         'train': '/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/features_embeddings/train/image_features.pth', 
    #         'valid': '/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/features_embeddings/valid/image_features.pth'
    #     },
    #     text_embedding_paths = {
    #         'train': '/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/features_embeddings/train/text_features.pth',
    #         'valid': '/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/features_embeddings/valid/text_features.pth'
    #     },
    #     reports_file_train = '/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
    #     reports_file_valid = '/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/valid_reports.csv',
    #     labels = "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv",
    #     results_folder="./checkpoints",
    #     batch_size = 3,
    #     num_train_steps = 100001,
    #     num_workers = 1, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
    #     train_from_scratch = True
    # )

    # uhn cluster
    trainer = CTClipTrainer(
        clip_xray,
        pretrained_xray_encoder = cfg_dot.training_params.use_pretrained_xray_encoder,
        min_epochs=cfg_dot.training_params.min_epochs,
        cfg=cfg,
        tokenizer=tokenizer,
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
        results_folder=f'/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/{xray_model_type}', # put the check point in a subdirectory under CT-RATE-CHECKPOINTS
        num_train_steps = 100001,
        batch_style=cfg_dot.training_params.batch_style,
        batch_size = cfg_dot.training_params.batch_size,
        num_workers = cfg_dot.training_params.num_workers, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
        train_from_scratch = cfg_dot.training_params.train_from_scratch,
        epoch_based_patience = cfg_dot.training_params.epoch_based_patience,
        iteration_evaluate_frequency = cfg_dot.training_params.iteration_evaluate_frequency,
        text_cl_weight = cfg_dot.training_params.text_cl_weight,
        ct_cl_weight = cfg_dot.training_params.ct_cl_weight, 
        # lr = 5e-3
        # cxr-clip parameters
        wd = cfg_dot.training_params.weight_decay,
        lr = cfg_dot.training_params.learning_rate
        # TODO: interpolate the learing rate between ULIP and CXR-CLIP
    )
    trainer.train_by_epoch(cfg_dot.training_params.epochs)

    """
    TODO: brainstorm different approachs for the contrastive learning function.
    """

if __name__ == '__main__':
    main()