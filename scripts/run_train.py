import os
from cxr_clip_utils import convert_dictconfig_to_dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch_geometric import seed_everything
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer, CTCLIPwithXray
from CTCLIPTrainer import CTClipTrainer

@hydra.main(
        version_base=None,
        config_path="C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\configs",
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

    seed_everything(1234)
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
        dim_xray = 768 if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False,
        cfg=cfg
    )

    # NOTE: get the related config file

    # NOTE: load the pretrained backbones
    ckpt_name = 'r50_mcc.tar' if cfg['model']['image_encoder']['name'] == 'resnet' else 'swint_mcc.tar'
    clip_xray.load("C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\Chris\\CT-CLIP\\models\\CT-CLIP_v2.pt",
                "C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\Chris\\CT-CLIP\\models\\cxr_clip\\{}".format(ckpt_name))

    """
    TODO: load the checkpoint for ct-clip (DONE)
    TODO: load the checkpoint for cxr-clip (DONE)
    TODO: tranformation config for cxr-clip (DONE)
    TODO: look for the dimension output from the image encoder of the xray clip (DONE)
    TODO: custom dataloader for triplet
    TODO: transformation of the input to the xray encoders
    TODO: ULIP-style loss function integration for cxr-clip and ct-clip

    TODO: double check the number of trainable parameters before and after freeze the ctclip model
    TODO: double check the orientation of the xray and the ct after processing.
    TODO: brainstorm different approachs for the contrastive learnining function.
    """

    trainer = CTClipTrainer(
        clip_xray,
        reports_file_train= "C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\dataset\\radiology_text_reports\\dataset_radiology_text_reports_validation_reports.csv",
        reports_file_valid= "C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\dataset\\radiology_text_reports\\dataset_radiology_text_reports_validation_reports.csv",
        data_train= "F:\\Chris\\dataset_preprocessed\\valid_preprocessed_ct",
        data_valid = "F:\\Chris\\dataset_preprocessed\\valid_preprocessed_ct",
        labels = "C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\dataset\\multi_abnormality_labels\\dataset_multi_abnormality_labels_valid_predicted_labels.csv",
        batch_size = 2,
        results_folder=".\\test",
        num_train_steps = 100001,
        num_workers = 4,
    )

    trainer.train()

if __name__ == '__main__':
    main()