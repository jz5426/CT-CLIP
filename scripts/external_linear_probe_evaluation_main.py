"""
do the same thing as in internal_linear_probe_evaluation_main and take that to evaluate on the mimic dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import CTReportDataSplitter, CTReportXRayClassificationDataset, MimicCTReportXRayDataset
from eval_utils import LinearProbeModel, XrayClassificationModel
from transformers import BertTokenizer, BertModel
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
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd
from zero_shot import CTClipInference
import shutil

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

    # print custom script argument
    print(f"is_linear_probe_eval: {cfg_dot.linear_probing_params.is_linear_probe_eval}")
    print(f"num_epochs: {cfg_dot.linear_probing_params.num_epochs}")
    print(f"patience: {cfg_dot.linear_probing_params.patience}")
    print(f"batch_size: {cfg_dot.linear_probing_params.batch_size}")
    print(f"learning_rate: {cfg_dot.linear_probing_params.learning_rate}")
    print(f"progress_window: {cfg_dot.linear_probing_params.progress_window}")
    print(f"cpt_dest: {cfg_dot.linear_probing_params.cpt_dest}")
    print(f"train data portion: {cfg_dot.linear_probing_params.train_data_portion}")

    # convert the config file to dictionary
    cfg = convert_dictconfig_to_dict(cfg_dot)

    torch.cuda.empty_cache()
    text_encoder = BertModel.from_pretrained(
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        local_files_only=True
    )
    tokenizer = BertTokenizer.from_pretrained(
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        do_lower_case=True,
        local_files_only=True)

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
    # cxr_clip_swin, cxr_clip_resnet, medclip_resnet, medclip_vit, gloria_densenet, gloria_resnet, custom_checkpoint_name
    if 'cxr_clip' in cfg_dot.linear_probing_params.baseline_type: # can be either cxr_clip_swin or cxr_clip_resnet
        xray_model_type = cfg_dot.linear_probing_params.baseline_type #'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
        dim_xray = 768 if 'swin' in cfg_dot.linear_probing_params.baseline_type else 2048  # if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048
        pth_base_name = 'swin_cxr_xray_features' if 'swin' in xray_model_type else 'resnet_cxr_xray_features'
    elif cfg_dot.linear_probing_params.baseline_type == 'medclip_resnet':
        xray_model_type = cfg_dot.linear_probing_params.baseline_type
        dim_xray = 2048
        pth_base_name = 'resnet_medclip_features'
        # place this somewhere in the medclip code to remove the learnt fc connected layer at the end, just like cxr_clip: del self.resnet.fc
    elif cfg_dot.linear_probing_params.baseline_type == 'medclip_vit':
        xray_model_type = cfg_dot.linear_probing_params.baseline_type
        dim_xray = 768
        pth_base_name = 'swin_medclip_features'
    elif cfg_dot.linear_probing_params.baseline_type == 'gloria_densenet':
        xray_model_type = cfg_dot.linear_probing_params.baseline_type
        dim_xray = 1024 #TODO: double check this.
        pth_base_name = 'densenet_gloria_features'
    elif cfg_dot.linear_probing_params.baseline_type == 'gloria_resnet':
        xray_model_type = cfg_dot.linear_probing_params.baseline_type
        dim_xray = 2048
        pth_base_name = 'resnet_gloria_features'
    else:
        xray_model_type = cfg_dot.linear_probing_params.baseline_type
        dim_xray = 768 if 'swin' in cfg_dot.linear_probing_params.baseline_type.lower() else 2048
        pth_base_name = f'{xray_model_type}_xray_features'

    #####
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
        auto_load_pretrained_weights=True # NOTE: automatically load the model weights based on the xray_model_type
    )

    pth_base_name = f'{pth_base_name}__train_portion_{cfg_dot.linear_probing_params.train_data_portion}'
    ckpt_parent_dir = os.path.join(cfg_dot.linear_probing_params.cpt_dest, 'mimic_ct')
    best_ckpt_destination = os.path.join(ckpt_parent_dir, f'{pth_base_name}_best_model.pth')

    #NOTE: train on the synthetic dataset and evaluate on the mimic-ct (256 images) data
    split = 'train'
    train_split_inference = CTClipInference(
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
        batch_size = 1024,
        num_train_steps = -1, # placeholder
        num_workers = 10, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
        feature_extraction_mode = True # might be optional
    )  

    pathologies = ['Arterial wall calcification', #
                    'Pericardial effusion', #
                    'Coronary artery wall calcification', #
                    'Hiatal hernia', #
                    'Lymphadenopathy', #
                    'Emphysema', #
                    'Atelectasis', #
                    'Mosaic attenuation pattern',#
                    'Peribronchial thickening', #
                    'Bronchiectasis', #
                    'Interlobular septal thickening']#

    # Initialize the wrapper model for either NOTE: linear probe or full model finetuninng
    # that is, add a additional fc layer on top of the vision model and the feature_projector
    num_classes = len(pathologies)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearProbeModel(in_features=latent_size, num_classes=num_classes)
    model.to(device)

    # get xray latent features from this particularly baseline model
    train_xray_features = train_split_inference.xray_feature_extraction('./', append=False)
    print('Xray feature extraction completed')

    # sanity check the trainable parameters
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of learnable parameters: {learnable_params}') # should be the same of a single linear layer

    # Set up the dataset and data loaders
    #NOTE: the label is the mimic version (with 11 labels) but the report and the data are the original CT-RATE
    train_data_splitter = CTReportDataSplitter(
        csv_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
        labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_mimic_labels.csv', #NOTE: the label need to be the mimic version
        data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/train_preprocessed_xray_mha',
    )
    train_sample, internal_val_samples = train_data_splitter.prepare_samples(
        train_split=cfg_dot.linear_probing_params.train_data_portion,
        val_split=0.2
    ) # validation split is always, train_split is controlable

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
        cfg=cfg,
        data=train_sample, # actual data potentially with the embeddings
        data_embeddings=train_xray_features,
        split='train'
    )

    internal_val_dataset = CTReportXRayClassificationDataset(
        cfg=cfg,
        data=internal_val_samples, # actual data potentially with the embeddings
        data_embeddings=train_xray_features,
        split='train'
    )

    #load the data
    train_loader = DataLoader(train_dataset, num_workers=cfg_dot.linear_probing_params.num_workers, batch_size=cfg_dot.linear_probing_params.batch_size, shuffle=True)
    val_loader = DataLoader(internal_val_dataset, num_workers=cfg_dot.linear_probing_params.num_workers, batch_size=cfg_dot.linear_probing_params.batch_size, shuffle=True)
    train_size = len(train_loader)

    # Training loop configuration
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg_dot.linear_probing_params.learning_rate)

    # Early stopping setup
    patience = cfg_dot.linear_probing_params.patience
    best_val_loss = float('inf')
    patience_counter = 0

    # NOTE: remove the all files under the 'mimic_ct' directory and resave it
    shutil.rmtree(ckpt_parent_dir, ignore_errors=True)

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
            os.makedirs(ckpt_parent_dir, exist_ok=True)
            torch.save(model.state_dict(), best_ckpt_destination)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Finetuning the Xray encoder completed ==> perform external mimic-ct testing")

    # testing
    # NOTE: use the mimic-ct external dataset to test it.
    test_dataset = MimicCTReportXRayDataset(
        cfg=cfg,
        data_folder='/cluster/home/t135419uhn/CT-CLIP/preprocessed_mimic/mimic_preprocessed_xray_mha',
        csv_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/external_valid_mimic_report.csv',
        labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_external_valid_mimic_labels.csv', 
        split='valid'
    )
    print(f'size of the external test data: {len(test_dataset)}')
    
    test_loader = DataLoader(
        test_dataset, 
        num_workers=cfg_dot.linear_probing_params.num_workers, 
        batch_size=cfg_dot.linear_probing_params.batch_size, 
        shuffle=False)

    # define a full classifier with encoder and projection layer learnt from the CT-RATE training set
    # NOTE: since mimic is a new dataset, so it requires full forward pass of the vision encoder and then the projection layer
    classification_model = XrayClassificationModel(
        vision_model=clip_xray.xray_encoder, 
        feature_projector=clip_xray.to_xray_latent, 
        pretrained_classifier=model, # load the classifier layer
        vision_model_type=xray_model_type
    )
    classification_model.to(device)

    test_params = {
        'test_loader': test_loader,
        'device': device,
        'model': classification_model,
        'pretrained_cpt_dest': best_ckpt_destination, # where to retrieve the best checkpoint
        'metric_saving_path': f'./lp_evaluation_results/mimic_ct/{pth_base_name}_test_metrics_results.xlsx' # where to save the files
    }
    test_loop(test_params)

    print('Finished probing evaluation without error :)')


def test_loop(params):

    test_loader = params['test_loader']
    device = params['device']
    model = params['model']
    metric_saving_path = params['metric_saving_path']
    
    all_labels = []
    all_preds = []
    all_probs = []

    # only load the pretrained classifier
    model.fc.load_state_dict(torch.load(params['pretrained_cpt_dest']))

    print(f'Performing testing with size (in unit batch) {len(test_loader)}')
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, _, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # For multilabel classification, apply sigmoid and threshold at 0.5
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            for i in range(labels.shape[0]):
                all_labels.append(labels[i,:].cpu().numpy())
                all_preds.append(preds[i,:].cpu().numpy())
                all_probs.append(probs[i,:].cpu().numpy())

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
    try:
        auc_macro = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
    except:
        auc_macro = -1
    try:
        auc_weighted = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
    except:
        auc_weighted = -1

    pr_auc_score_micro = average_precision_score(all_labels, all_probs, average='micro')
    pr_auc_score_macro = average_precision_score(all_labels, all_probs, average='macro')
    pr_auc_score_weighted = average_precision_score(all_labels, all_probs, average='weighted')

    print(f"Test Results for micro average: F1 Score: {f1_micro:.4f}, Recall: {recall_micro:.4f}, Precision: {precision_micro:.4f}, AUC: {auc_micro:.4f}, PR_AUC: {pr_auc_score_micro:.4f}")
    print(f"Test Results for weighted average: F1 Score: {f1_weighted:.4f}, Recall: {recall_weighted:.4f}, Precision: {precision_weighted:.4f}, AUC: {auc_weighted:.4f}, PR_AUC: {pr_auc_score_weighted:.4f}")
    print(f"Test Results for macro average: F1 Score: {f1_macro:.4f}, Recall: {recall_macro:.4f}, Precision: {precision_macro:.4f}, AUC: {auc_macro:.4f}, PR_AUC: {pr_auc_score_macro:.4f}")

    print('Saving the metrics results')
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1 Score', 'AUC', 'PR_AUC', 'labels', 'pred_probs'], # the labels and the pred_probs are for delong auc significant test
        'Micro': [precision_micro, recall_micro, f1_micro, auc_micro, pr_auc_score_micro, all_labels.flatten().tolist(), all_probs.flatten().tolist()],
        'Weighted': [precision_weighted, recall_weighted, f1_weighted, auc_weighted, pr_auc_score_weighted, -1, -1],
        'Macro': [precision_macro, recall_macro, f1_macro, auc_macro, pr_auc_score_macro, -1, -1]
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