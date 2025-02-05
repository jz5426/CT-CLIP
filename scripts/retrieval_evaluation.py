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
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from zero_shot import CTClipInference
from retrieval_evaluation_utils import recall_retrieval_evaluation, map_retrieval_evaluation
import pandas as pd

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
    #dim_image = 131072,

    print('Starting Xray related retrieval experiments')

    # our retrival results: from cxr_clip model, from our pretrained xray encoder distilled from ct_clip
    ckpt_names = [
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
    for ckpt_name in ckpt_names:

        #NOTE cfg is mainly for cxr_clip
        if 'cxr_clip' in ckpt_name: # can be either cxr_clip_swin or cxr_clip_resnet
            xray_model_type = ckpt_name #'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
            dim_xray = 768 if 'swin' in ckpt_name else 2048  # if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048
            pth_name = 'swin_cxr_xray_features.pth' if 'swin' in ckpt_name else 'resnet_cxr_xray_features.pth'
        elif ckpt_name == 'medclip_resnet':
            xray_model_type = ckpt_name
            dim_xray = 2048
            pth_name = 'resnet_medclip_features.pth'

            # place this somewhere in the medclip code to remove the learnt fc connected layer at the end, just like cxr_clip: del self.resnet.fc
        elif ckpt_name == 'medclip_vit':
            xray_model_type = ckpt_name
            dim_xray = 768
            pth_name = 'swin_medclip_features.pth'
        
        # elif ckpt_name == 'gloria_densenet':
        #     xray_model_type = ckpt_name
        #     dim_xray = 1024 #TODO: double check this.
        #     pth_name = 'densenet_gloria_features.pth'

        # elif ckpt_name == 'gloria_resnet':
        #     xray_model_type = ckpt_name
        #     dim_xray = 2048
        #     pth_name = 'resnet_gloria_features.pth'

        else:
            # our pretrained model
            xray_model_type = ckpt_name
            dim_xray = 768 if 'swin' in ckpt_name.lower() else 2048
            pth_name = f'{ckpt_name}_xray_features.pth'

        # automatically load the model weights
        clip_xray = CTCLIPwithXray(
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            dim_text = 768,
            dim_image = 294912,
            xray_model_type = xray_model_type,
            dim_xray = dim_xray,
            dim_latent = 512,
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False,
            cfg=cfg
        )

        # check the trainable parameters
        # xray_encoder_trainable = sum(p.numel() for p in clip_xray.xray_encoder.parameters() if p.requires_grad)
        # ct_clip_trainable = sum(p.numel() for p in clip_xray.CTCLIP.parameters() if p.requires_grad)
        # assert(xray_encoder_trainable == 0)
        # assert(ct_clip_trainable == 0)

        retrival_evaluator = CTClipInference(
            clip_xray,
            cfg=cfg,
            tokenizer=tokenizer,
            data_folder= f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/{split}_preprocessed_xray_mha',
            # NOTE: the embedding paths are MANDATORY for the dataloader to work. RUN THIS SCRIPT MAINLY AFTER THE CTCLIP EMBEDDINGS ARE EXTRACTED.
            img_embedding_paths = {
                f'{split}': f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/{split}/image_features.pth'
            },
            text_embedding_paths = {
                f'{split}': f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/{split}/text_features.pth'
            },
            reports_file = f'/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/{split}_reports.csv',
            labels = f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            results_folder="./inference_zeroshot_retrieval",
            batch_size = 512,
            num_train_steps = -1, # placeholder
            num_workers = 10, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
            feature_extraction_mode = True # might be optional
        )  

        # get xray latent features from a model NOTE: to be safe, re-extract the xray feature everytime
        xray_features = retrival_evaluator.xray_feature_extraction(embedding_directory, pth_name=pth_name, append=False)

        # make sure all three dictionary contains the same set of keys
        assert(image_features.keys() == text_features.keys() == xray_features.keys())

        # organize data into a list with index as a the text-image-xray correspondance and pair up xray-ct_image and xray-text
        triplet_embeddings = [(image_features[key], text_features[key], xray_features[key]) for key in xray_features.keys()]

        # NOTE: all features are normalized.

        print('evaluating xray 2 ct_volumes recall')
        recall_retrieval_evaluation(
            query_latents=[triple[-1] for triple in triplet_embeddings],
            target_latents=[triple[0].reshape(-1) for triple in triplet_embeddings],
            file_name=f'{ckpt_name}_synxray2ct_recall')
        print('evaluating ct_volumes 2 xray recall')
        recall_retrieval_evaluation(
            query_latents=[triple[0] for triple in triplet_embeddings],
            target_latents=[triple[-1].reshape(-1) for triple in triplet_embeddings],
            file_name=f'{ckpt_name}_ct2synxray_recall')

        print('evaluating xray 2 ct_reports recall')
        recall_retrieval_evaluation(
            query_latents=[triple[-1] for triple in triplet_embeddings],
            target_latents=[triple[1].reshape(-1) for triple in triplet_embeddings],
            file_name=f'{ckpt_name}_synxray2report_recall')
        print('evaluating ct_reports 2 xray recall')
        recall_retrieval_evaluation(
            query_latents=[triple[1] for triple in triplet_embeddings],
            target_latents=[triple[-1].reshape(-1) for triple in triplet_embeddings],
            file_name=f'{ckpt_name}_report2synxray_recall')



        print('evaluating xray 2 ct_volumes MAP')
        map_retrieval_evaluation(
            xray_features,
            target_latents=image_features,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{ckpt_name}_synxray2ct_map')
        print('evaluating ct_volumes 2 xray MAP')
        map_retrieval_evaluation(
            image_features,
            target_latents=xray_features,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{ckpt_name}_ct2synxray_map')



        print('evaluating xray 2 ct_reports MAP')
        map_retrieval_evaluation(
            xray_features,
            target_latents=text_features,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{ckpt_name}_synxray2report_map')
        print('evaluating ct_reports 2 xray MAP')
        map_retrieval_evaluation(
            text_features,
            target_latents=xray_features,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{ckpt_name}_report2synxray_map')



        # there is not symmetric retrieval and recall for this one.
        print('evaluating xray 2 xray MAP')
        map_retrieval_evaluation(
            xray_features,
            target_latents=xray_features,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{ckpt_name}_synxray2synxray_map')

if __name__ == '__main__':

    main()