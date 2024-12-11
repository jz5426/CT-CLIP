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

    # windows
    # clip_xray.load("C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\Chris\\CT-CLIP\\models\\CT-CLIP_v2.pt",
    #             "C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\Chris\\CT-CLIP\\models\\cxr_clip\\{}".format(ckpt_name))

    # windows wsl
    # clip_xray.load("/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/models/CT-CLIP_v2.pt",
    #             "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/models/cxr_clip/{}".format(ckpt_name))

    # uhn cluster
    clip_xray.load("/cluster/home/t135419uhn/CT-CLIP/models/CT-CLIP_v2.pt",
                "/cluster/home/t135419uhn/CT-CLIP/models/cxr_clip/{}".format(ckpt_name))

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
    #     batch_size = 3,
    #     results_folder="./checkpoints",
    #     num_train_steps = 100001,
    #     num_workers = 1, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
    #     train_from_scratch = True
    # )

    # # uhn cluster
    trainer = CTClipTrainer(
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
        num_train_steps = 100001,
        num_workers = 10, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
        train_from_scratch = True,
        epoch_based_patience = 10,
        iteration_evaluate_frequency = 4,
        text_cl_weight = 1.,
        ct_cl_weight = 1., 
        # lr = 5e-3
        # cxr-clip parameters
        wd = 1e-4,
        lr = 5e-5
        # TODO: interpolate the learing rate between ULIP and CXR-CLIP
    )

    trainer.train_by_epoch(500)
    """
    TODO: check the performance when the xray encoder is initialized from scratch.
    TODO: brainstorm different approachs for the contrastive learning function.
    """

if __name__ == '__main__':
    main()