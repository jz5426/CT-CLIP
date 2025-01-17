"""
main evaluation training loop for finetune/linear probe the xray encoder only.

stratification for multilabel dataset:
- https://github.com/trent-b/iterative-stratification
- http://scikit.ml/stratification.html
- https://datascience.stackexchange.com/questions/45174/how-to-use-sklearn-train-test-split-to-stratify-data-for-multi-label-classificat

- think of the following script as linear probing with 10%, 20%, 50%, 80%, 100% of the data and see the performance comparison.
- follow the "make the most of the text semantics to improve biomedical VLP", try percentage 0, 1, 10, 100 percent of data during linear probing
# TODO: do the following
    - performance in auroc, f1, flat acc, precision for 0, 1, 10, 100 during linear probe in multilabel classification. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import CTReportDataSplitter, CTReportXRayClassificationDataset
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
        config_path="/cluster/home/t135419uhn/CT-CLIP/configs",
        # config_path="/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/configs",
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

    # print custom script argument
    print(f"use_binary_classification: {cfg_dot.linear_probing_params.use_binary_classification}")
    print(f"is_linear_probe_eval: {cfg_dot.linear_probing_params.is_linear_probe_eval}")
    print(f"is_evaluate_our_model: {cfg_dot.linear_probing_params.is_evaluate_our_model}")
    print(f"num_epochs: {cfg_dot.linear_probing_params.num_epochs}")
    print(f"patience: {cfg_dot.linear_probing_params.patience}")
    print(f"batch_size: {cfg_dot.linear_probing_params.batch_size}")
    print(f"learning_rate: {cfg_dot.linear_probing_params.learning_rate}")
    print(f"progress_window: {cfg_dot.linear_probing_params.progress_window}")
    print(f"cpt_dest: {cfg_dot.linear_probing_params.cpt_dest}")

    # convert the config file to dictionary
    cfg = convert_dictconfig_to_dict(cfg_dot)

    torch.cuda.empty_cache()
    # text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder = BertModel.from_pretrained(
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        local_files_only=True
    ) #TODO: uncomment this when not testing
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
    if cfg_dot.linear_probing_params.is_evaluate_our_model:
        ckp_name = 'modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch'
        clip_xray.load_pretrained_ct_xray_clip(f'/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/{ckp_name}.pt')
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
    num_classes = 1 if cfg_dot.linear_probing_params.use_binary_classification else len(pathologies)
    model = XrayClassificationModel(
        vision_model=clip_xray.xray_encoder, 
        feature_projector=clip_xray.to_xray_latent, 
        isLinearProbe=cfg_dot.linear_probing_params.is_linear_probe_eval, 
        in_features=latent_size, 
        num_classes=num_classes)

    # sanity check the trainable parameters
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of learnable parameters: {learnable_params}') # should be the same of a single linear layer

    # Set up the dataset and data loaders
    #TODO: allow only train with a percentage of data, changes the .sample in the dataset and put that back in.
    train_data_splitter = CTReportDataSplitter(
        csv_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
        labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv',
        data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/train_preprocessed_xray_mha',
    )
    train_sample, internal_val_samples = train_data_splitter.prepare_samples(split_percentage=0.2)

    # sanity check
    # Extract multi-hot label vectors from the samples
    # train_labels = np.array([label for _, label in train_sample])
    # val_labels = np.array([label for _, label in internal_val_samples])
    # train_label_distribution = train_labels.sum(axis=0)
    # val_label_distribution = val_labels.sum(axis=0)
    # print(train_label_distribution)
    # print(val_label_distribution)
    # print(val_label_distribution/train_label_distribution)

    train_dataset = CTReportXRayClassificationDataset(
        # data_folder='/mnt/g/Chris/CT-RATE-FINAL/processed_dataset/train_preprocessed_xray_mha', # data path for the xray train
        # report_file='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
        # labels='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv' # path for train xray data label
        # data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/train_preprocessed_xray_mha',
        # report_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
        # labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv',
        cfg=cfg,
        data=train_sample, # actual data
        split='train'
    )

    internal_val_dataset = CTReportXRayClassificationDataset(
        # data_folder='/mnt/g/Chris/CT-RATE-FINAL/processed_dataset/train_preprocessed_xray_mha', # data path for the xray train
        # report_file='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
        # labels='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv' # path for train xray data label
        # data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/train_preprocessed_xray_mha',
        # report_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
        # labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv',
        cfg=cfg,
        data=internal_val_samples, # actual data
        split='train'
    )

    test_data_splitter = CTReportDataSplitter(
        csv_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/valid_reports.csv',
        labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/valid_preprocessed_xray_mha',
    )
    test_samples = test_data_splitter.prepare_samples(split_percentage=1.) # no splitting

    test_dataset = CTReportXRayClassificationDataset(
        # data_folder='/mnt/g/Chris/CT-RATE-FINAL/processed_dataset/valid_preprocessed_xray_mha', # data path for the xray val
        # report_file='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/valid_reports.csv',
        # labels='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv' # path for val xray data label
        # data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/valid_preprocessed_xray_mha',
        # report_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/valid_reports.csv',
        # labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        cfg=cfg,
        data=test_samples,
        split='valid'
    )

    #load the data
    train_loader = DataLoader(train_dataset, batch_size=cfg_dot.linear_probing_params.batch_size, shuffle=True)
    val_loader = DataLoader(internal_val_dataset, batch_size=cfg_dot.linear_probing_params.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg_dot.linear_probing_params.batch_size, shuffle=False)
    train_size, val_size, test_size = len(train_loader), len(val_loader), len(test_loader)

    # Training loop configuration
    criterion = nn.BCEWithLogitsLoss() if not cfg_dot.linear_probing_params.use_binary_classification else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg_dot.linear_probing_params.learning_rate)

    # Early stopping setup
    patience = cfg_dot.linear_probing_params.patience
    best_val_loss = float('inf')
    patience_counter = 0

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training and validation loop
    for epoch in range(cfg_dot.linear_probing_params.num_epochs):
        model.train()
        total_loss = 0.0

        for idx, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels.float() if not cfg_dot.linear_probing_params.use_binary_classification else labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if idx % cfg_dot.linear_probing_params.progress_window == 0:
                print(f"Epoch [{epoch}/{cfg_dot.linear_probing_params.num_epochs}], Batch [{idx}/{train_size}] in training split, Training Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{cfg_dot.linear_probing_params.num_epochs}, Training Loss: {total_loss/len(train_loader):.4f}")

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
                loss = criterion(outputs, labels.float() if not cfg_dot.linear_probing_params.use_binary_classification else labels)
                val_loss += loss.item()

            if idx % cfg_dot.linear_probing_params.progress_window == 0:
                print(f"Epoch [{epoch}/{cfg_dot.linear_probing_params.num_epochs}], Batch [{idx}/{val_size}] in training split, Validation Loss: {loss.item():.4f}")

        val_loss /= len(test_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(cfg_dot.linear_probing_params.cpt_dest, exist_ok=True)
            dest = os.path.join(cfg_dot.linear_probing_params.cpt_dest, f'{pth_base_name}_best_model.pth')
            torch.save(model.state_dict(), dest)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Finetuning the XRay encoder complete. ==> perform internal testing")

    #TODO: evaluate the model on the test set to find the auc, recall, and precision metrics

# def parse_args():
#     parser = argparse.ArgumentParser(description="Train Vision Model Wrapper")
#     parser.add_argument("--use_binary_classification", type=bool, default=False,  help="Toggle for binary classification")
#     parser.add_argument("--is_linear_probe_eval", type=bool, default=True, help="linear probing evaluation or full model finetuning")
#     parser.add_argument("--is_evaluate_our_model", type=bool, default=True, help="whether evalute our model or the one from cxr_clip")
#     parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
#     parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
#     parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
#     parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
#     parser.add_argument("--progress_window", type=int, default=2, help="show progress every progress window elapsed")
#     parser.add_argument('--cpt_dest', type=str, default='/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/classification_evaluation', help='destinatin folder for the weights')
#     args = parser.parse_args()

#     return args

# Example usage
if __name__ == "__main__":
    main()