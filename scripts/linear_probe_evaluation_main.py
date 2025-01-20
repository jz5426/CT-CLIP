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
    - NOTE this script is missing zero-shot performance evaluation.
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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import pandas as pd

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

    if cfg_dot.linear_probing_params.is_evaluate_our_model:
        # ckp_name = 'modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch'
        ckp_name = cfg_dot.linear_probing_params.ckpt_name #TODO:
        clip_xray.load_pretrained_ct_xray_clip(f'/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/{ckp_name}.pt')
        pth_base_name = f'{ckp_name}_pretrained_xray_encoder_features'
    else:
        # evalute the model from cxr_clip
        ckpt_name = 'r50_mcc.tar' if cfg['model']['image_encoder']['name'] == 'resnet' else 'swint_mcc.tar'
        clip_xray.load_xray_encoder(
            '/cluster/home/t135419uhn/CT-CLIP/models/cxr_clip/{}'.format(ckpt_name), # cxr-clip pretrained
            freeze_weights=True
        )
        pth_base_name = 'cxr_clip_pretrained_xray_encoder_features'

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
    num_classes = len(pathologies)
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
    train_loader = DataLoader(train_dataset, num_workers=cfg_dot.linear_probing_params.num_workers, batch_size=cfg_dot.linear_probing_params.batch_size, shuffle=True)
    val_loader = DataLoader(internal_val_dataset, num_workers=cfg_dot.linear_probing_params.num_workers, batch_size=cfg_dot.linear_probing_params.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,num_workers=cfg_dot.linear_probing_params.num_workers, batch_size=cfg_dot.linear_probing_params.batch_size, shuffle=False)
    train_size, val_size, test_size = len(train_loader), len(val_loader), len(test_loader)

    # Training loop configuration
    criterion = nn.BCEWithLogitsLoss()
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
        
        # train loop
        train_params = {
            'train_loader': train_loader,
            'device': device,
            'model': model,
            'criterion': criterion,
            'optimizer': optimizer,
            'epoch': epoch,
            'train_size': train_size,
            'progress_window': cfg_dot.linear_probing_params.progress_window,
            'num_epochs': cfg_dot.linear_probing_params.num_epochs
        }
        train_loop(train_params)

        # Validation loop
        val_params = {
            'val_loader': val_loader,
            'device': device,
            'model': model,
            'criterion': criterion,
        }
        val_loss = validation_loop(val_params)

        # early stopping mechanism
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

    print("Finetuning the Xray encoder completed ==> perform internal testing")

    # testing
    test_params = {
        'test_loader': test_loader,
        'device': device,
        'model': model,
        'pretrained_cpt_dest': os.path.join(cfg_dot.linear_probing_params.cpt_dest, f'{pth_base_name}_best_model.pth'),
        'metric_saving_path': f'./lp_evaluation_results/{pth_base_name}_internal_test_metrics_results.xlsx'
    }
    test_loop(test_params)

    print('Finished probing evaluation without error :)')


def test_loop(params):
    test_loader = params['test_loader']
    device = params['device']
    model = params['model']
    metric_saving_path = params['metric_saving_path']
    
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    # load the pretrained model TODO: uncomment when finished testing
    model.load_state_dict(torch.load(params['pretrained_cpt_dest']))

    print(f'Performing testing with size (in unit batch) {len(test_loader)}')
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # For multilabel classification, apply sigmoid and threshold at 0.5
            # TODO: double check this.
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays for metric computation
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate metrics for multilabel classification
    # NOTE: might use the same one from the training file instead of using the sklearn one.
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    auc_micro = roc_auc_score(all_labels, all_probs, average='micro', multi_class='ovr')
    auc_weighted = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
    auc_macro = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')

    print(f"Test Results for micro average: Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1 Score: {f1_micro:.4f}, AUC: {auc_micro:.4f}")
    print(f"Test Results for weighted average: Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1 Score: {f1_weighted:.4f}, AUC: {auc_weighted:.4f}")
    print(f"Test Results for macro average: Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1 Score: {f1_macro:.4f}, AUC: {auc_macro:.4f}")

    print('Saving the metrics results')
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1 Score', 'AUC'],
        'Micro': [precision_micro, recall_micro, f1_micro, auc_micro],
        'Weighted': [precision_weighted, recall_weighted, f1_weighted, auc_weighted],
        'Macro': [precision_macro, recall_macro, f1_macro, auc_macro]
    }

    metrics_df = pd.DataFrame(metrics_data)
    os.makedirs(os.path.dirname(metric_saving_path), exist_ok=True)
    metrics_df.to_excel(metric_saving_path, index=False)
    print(f"Metric results saved to {metric_saving_path}")


def validation_loop(params):
    val_loader = params['val_loader']
    device = params['device']
    model = params['model']
    criterion = params['criterion']

    model.eval()
    val_loss = 0.0
    print(f'Performing validation with size (in unit batch) {len(val_loader)}')
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

    return val_loss

def train_loop(params):
    train_loader = params['train_loader']
    device = params['device']
    model = params['model']
    criterion = params['criterion']
    optimizer = params['optimizer']
    epoch = params['epoch']
    train_size = params['train_size']
    progress_window = params['progress_window']
    num_epochs = params['num_epochs']
    
    model.train()
    total_loss = 0.0
    for idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if idx % progress_window == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{idx}/{train_size}] in training split, Training Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss/len(train_loader):.4f}")


# Example usage
if __name__ == "__main__":
    main()