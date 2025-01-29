import os
from cxr_clip_utils import convert_dictconfig_to_dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
# from torch_geometric import seed_everything
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIPwithXray
from CTCLIPTrainer import CTClipTrainer
import random
import numpy as np

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

    if 'cxr_clip' in cfg_dot.training_params.training_pretrain_baseline: # can be either cxr_clip_swin or cxr_clip_resnet
        xray_model_type = cfg_dot.training_params.training_pretrain_baseline #'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
        dim_xray = 768 if 'swin' in cfg_dot.training_params.training_pretrain_baseline else 2048  # if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048
    elif cfg_dot.training_params.training_pretrain_baseline == 'medclip_resnet':
        xray_model_type = cfg_dot.training_params.training_pretrain_baseline
        dim_xray = 2048
        # place this somewhere in the medclip code to remove the learnt fc connected layer at the end, just like cxr_clip: del self.resnet.fc
    elif cfg_dot.training_params.training_pretrain_baseline == 'medclip_vit':
        xray_model_type = cfg_dot.training_params.training_pretrain_baseline
        dim_xray = 768
    elif cfg_dot.training_params.training_pretrain_baseline == 'gloria_densenet':
        xray_model_type = cfg_dot.training_params.training_pretrain_baseline
        dim_xray = 1024 #TODO: double check this.
    elif cfg_dot.training_params.training_pretrain_baseline == 'gloria_resnet':
        xray_model_type = cfg_dot.training_params.training_pretrain_baseline
        dim_xray = 2048
    else:
        xray_model_type = cfg_dot.training_params.training_pretrain_baseline
        dim_xray = 768 if 'swin' in cfg_dot.training_params.training_pretrain_baseline.lower() else 2048


    # xray_model_type = 'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
    # clip_xray = CTCLIPwithXray(
    #     image_encoder = image_encoder,
    #     text_encoder = text_encoder,
    #     # tokenizer=tokenizer,
    #     dim_text = 768,
    #     dim_image = 294912,
    #     xray_model_type = xray_model_type,
    #     dim_xray = 768 if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048,
    #     dim_latent = 512,
    #     extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    #     use_mlm=False,
    #     downsample_image_embeds = False,
    #     use_all_token_embeds = False,
    #     cfg=cfg,
    #     auto_load_pretrained_weights=False # because it loads it later.
    # )

    # # uhn cluster
    # #NOTE: if cfg_dot.training_params.use_pretrained_xray_encoder is true => xray encoder and the projection layer is loaded with pretrained cxr_clip weights
    # # Load the CT-CLIP pretrained backbone to CT-CLIP and optionlly load the pretrained cxr_clip xray encoder weights
    # ckpt_name = 'r50_mcc.tar' if cfg['model']['image_encoder']['name'] == 'resnet' else 'swint_mcc.tar' # NOTE: weights for cxr_clip xray encoder
    # clip_xray.load(
    #     "/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/CT-CLIP_v2.pt",
    #     f"/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/cxr_clip/{ckpt_name}" if cfg_dot.training_params.use_pretrained_xray_encoder else None
    # )

    # for custom pretrained weight training
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
        auto_load_pretrained_weights=True if cfg_dot.training_params.use_pretrained_xray_encoder else False,
        freeze_xray_pretrained_weights=False # need the xray encoder for training => no freeze parameters in xray encoder
    )
    # load the ct-clip pretrained weights
    clip_xray.load_ctclip('/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/CT-CLIP_v2.pt')

    # check the trainable parameters
    xray_encoder_trainable = sum(p.numel() for p in clip_xray.xray_encoder.parameters() if p.requires_grad)
    ct_clip_trainable = sum(p.numel() for p in clip_xray.CTCLIP.parameters() if p.requires_grad)
    assert(xray_encoder_trainable > 0)
    assert(ct_clip_trainable == 0)

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
        lr = cfg_dot.training_params.learning_rate,
        model_type=xray_model_type
        # TODO: interpolate the learing rate between ULIP and CXR-CLIP
    )
    trainer.train_by_epoch(cfg_dot.training_params.epochs)

    """
    TODO: brainstorm different approachs for the contrastive learning function.
    """

if __name__ == '__main__':
    main()