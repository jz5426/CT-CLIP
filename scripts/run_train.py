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

@hydra.main(
        version_base=None,
        # config_path="C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\configs",
        config_path="/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/configs",
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

    run(cfg)


def run(cfg):
        
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

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
    # clip_xray.load("C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\Chris\\CT-CLIP\\models\\CT-CLIP_v2.pt",
    #             "C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\Chris\\CT-CLIP\\models\\cxr_clip\\{}".format(ckpt_name))

    clip_xray.load("/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/models/CT-CLIP_v2.pt",
                "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/models/cxr_clip/{}".format(ckpt_name))

    # check the trainable parameters
    xray_encoder_trainable = sum(p.numel() for p in clip_xray.xray_encoder.parameters() if p.requires_grad)
    ct_clip_trainable = sum(p.numel() for p in clip_xray.CTCLIP.parameters() if p.requires_grad)
    assert(xray_encoder_trainable > 0)
    assert(ct_clip_trainable == 0)

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

    trainer = CTClipTrainer(
        clip_xray,
        cfg=cfg,
        data_train= "/mnt/f/Chris/dataset/train_preprocessed_xray_mha",
        data_valid = "/mnt/f/Chris/dataset/valid_preprocessed_xray_mha",
        labels = "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv",
        batch_size = 2,
        results_folder="./checkpoints",
        num_train_steps = 100001,
        num_workers = 1, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
        train_from_scratch = True
    )

    ##trainer.train() # train by iterations
    trainer.train_by_epoch(250)

    """
    TODO: check the performance when the xray encoder is initialized from scratch.
    TODO: brainstorm different approachs for the contrastive learning function.
    """

if __name__ == '__main__':
    main()