"""
main evaluation training loop for finetune/linear probe the xray encoder only.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from data import CTReportXRayClassificationDataset
from eval_utils import XrayClassificationModel

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
from torch.utils.data import DataLoader


@hydra.main(
        version_base=None,
        # config_path="/cluster/home/t135419uhn/CT-CLIP/configs",
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

    # seed everything
    seed = 1024
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    run(cfg)

def run(cfg):

    # custom script argument
    args = parse_args()

    torch.cuda.empty_cache()
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    # text_encoder = BertModel.from_pretrained(
    #     '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
    #     local_files_only=True
    # ) #TODO: uncomment this when not testing
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

    latent_size = 512
    clip_xray = CTCLIPwithXray(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_text = 768,
        dim_image = 294912,
        xray_model_type = 'swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'resnet',
        dim_xray = 768 if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048, # output size of the xray feature extractor
        dim_latent = latent_size, # latent size that match the CT vision encoder and the text encoder.
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False,
        cfg=cfg
    )

    #TODO: uncomment this when not testing.
    if args.is_evaluate_our_model:
        ckp_name = 'CTClip_lowest_val_cl_loss_during_iterations'
        # clip_xray.load_pretrained_ct_xray_clip(f'/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/{ckp_name}.pt')
        pth_base_name = f'{ckp_name}_pretrained_xray_encoder_features'
    else:
        # evalute the model from cxr_clip
        ckpt_name = 'r50_mcc.tar' if cfg['model']['image_encoder']['name'] == 'resnet' else 'swint_mcc.tar'
        clip_xray.load_xray_encoder(
            '/cluster/home/t135419uhn/CT-CLIP/models/cxr_clip/{}'.format(ckpt_name), # cxr-clip pretrained
            freeze_weights=True
        )
        pth_base_name = 'cxr_clip_pretrained_xray_encoder_features.pth'

    # modify the base file name

    pathologies = ['Medical material',
                    'Arterial wall calcification', 
                    'Cardiomegaly', 
                    'Pericardial effusion',
                    'Coronary artery wall calcification', 
                    'Hiatal hernia',
                    'Lymphadenopathy', 
                    'Emphysema', 
                    'Atelectasis', 
                    'Lung nodule',
                    'Lung opacity', 
                    'Pulmonary fibrotic sequela', 
                    'Pleural effusion', 
                    'Mosaic attenuation pattern',
                    'Peribronchial thickening', 
                    'Consolidation', 
                    'Bronchiectasis',
                    'Interlobular septal thickening']

    # Initialize the wrapper model for either NOTE: linear probe or full model finetuninng
    num_classes = 1 if args.use_binary_classification else len(pathologies)
    model = XrayClassificationModel(
        vision_model=clip_xray.xray_encoder, 
        feature_projector=clip_xray.to_xray_latent, 
        isLinearProbe=args.is_linear_probe_eval, 
        in_features=latent_size, 
        num_classes=num_classes)

    # sanity check the trainable parameters
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of learnable parameters: {learnable_params}')

    # Set up the dataset and data loaders
    train_dataset = CTReportXRayClassificationDataset(
        # data_folder='/mnt/g/Chris/CT-RATE-FINAL/processed_dataset/train_preprocessed_xray_mha', # data path for the xray train
        # report_file='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
        # labels='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv' # path for train xray data label
        data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/train_preprocessed_xray_mha',
        report_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
        labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv',
        cfg=cfg,
    )
    val_dataset = CTReportXRayClassificationDataset(
        # data_folder='/mnt/g/Chris/CT-RATE-FINAL/processed_dataset/valid_preprocessed_xray_mha', # data path for the xray val
        # report_file='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/valid_reports.csv',
        # labels='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv' # path for val xray data label
        data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/valid_preprocessed_xray_mha',
        report_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/valid_reports.csv',
        labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        cfg=cfg,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    train_size = len(train_loader)
    val_size = len(val_loader)

    # Training loop configuration
    criterion = nn.BCEWithLogitsLoss() if not args.use_binary_classification else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Early stopping setup
    patience = args.patience
    best_val_loss = float('inf')
    patience_counter = 0

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training and validation loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for idx, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels.float() if not args.use_binary_classification else labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if idx % args.progress_window == 0:
                print(f"Epoch [{epoch}/{args.num_epochs}], Batch [{idx}/{train_size}] in training split, Training Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {total_loss/len(train_loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels.float() if not args.use_binary_classification else labels)
                val_loss += loss.item()

            if idx % args.progress_window == 0:
                print(f"Epoch [{epoch}/{args.num_epochs}], Batch [{idx}/{val_size}] in training split, Validation Loss: {loss.item():.4f}")

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(args.cpt_dest, exist_ok=True)
            dest = os.path.join(args.cpt_dest, f'{pth_base_name}_best_model.pth')
            torch.save(model.state_dict(), dest)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Finetuning the XRay encoder complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Model Wrapper")
    parser.add_argument("--use_binary_classification", type=bool, default=False,  help="Toggle for binary classification")
    parser.add_argument("--is_linear_probe_eval", type=bool, default=True, help="linear probing evaluation or full model finetuning")
    parser.add_argument("--is_evaluate_our_model", type=bool, default=True, help="whether evalute our model or the one from cxr_clip")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--progress_window", type=int, default=2, help="show progress every progress window elapsed")
    parser.add_argument('--cpt_dest', type=str, default='/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/classification_evaluation', help='destinatin folder for the weights')
    args = parser.parse_args()

    return args

# Example usage
if __name__ == "__main__":
    main()