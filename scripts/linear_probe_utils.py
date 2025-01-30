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
import pickle

def load_cached_ct_rate_xray_features(split, clip_xray, cfg, tokenizer):
    # TODO: should load from cache
    # split = 'train'
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

    # get xray latent features from this particularly baseline model
    train_xray_features = train_split_inference.xray_feature_extraction('./', append=False)
    print('Xray feature extraction completed')

    return train_xray_features

def get_train_internal_split(dataset='mimic'):
    """
    mimic and internal evalution share the same strategy

    set it up so that it loads from cache instead of loading from scratch
    """
    if dataset == 'mimic' or dataset == 'internal':
        train_data_splitter = CTReportDataSplitter(
            csv_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
            labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_mimic_labels.csv', #NOTE: the label need to be the mimic version
            data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/train_preprocessed_xray_mha',
        )
        train_sample, internal_val_samples = train_data_splitter.prepare_samples(
            train_split=cfg_dot.linear_probing_params.train_data_portion,
            val_split=0.2
        ) # validation split is always, train_split is controlable

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

    return train_dataset, internal_val_dataset


def get_pathologies(dataset='internal'):
    if dataset == 'internal':
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
    elif dataset == 'mimic':
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
    elif dataset == 'vinBig':
        pathologies = []
        # TODO:

    return pathologies


def linear_probing_main():
    # Initialize the wrapper model for either NOTE: linear probe or full model finetuninng
    # that is, add a additional fc layer on top of the vision model and the feature_projector
    num_classes = len(pathologies)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearProbeModel(in_features=latent_size, num_classes=num_classes)
    model.to(device)

    # sanity check the trainable parameters
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of learnable parameters: {learnable_params}') # should be the same of a single linear layer

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


def evaluate_classifier(dataset='', model):
    if dataset == 'mimic':
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
            'metric_saving_path': f'./lp_evaluation_results/mimic_ct/{pth_base_name}_test_metrics_results.xlsx', # where to save the files
            'delong_stats_saving_path': f'./lp_evaluation_results/mimic_ct/delong_stats/{pth_base_name}_data.pkl'
        }
        test_loop(test_params)
    elif dataset == 'internal':
        val_xray_features = load_cached_ct_rate_xray_features(split='valid') #TODO:
        print('Xray feature extraction completed on the validation split for this particular baseline model')

        # the whole validation dataset for internal validation.
        test_data_splitter = CTReportDataSplitter(
            csv_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/valid_reports.csv',
            labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
            data_folder='/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/valid_preprocessed_xray_mha',
        )
        test_samples = test_data_splitter.prepare_samples(train_split=1., val_split=0.) # no splitting

        test_dataset = CTReportXRayClassificationDataset(
            cfg=cfg,
            data=test_samples,
            data_embeddings=val_xray_features,
            split='valid'
        )
        print(f'size of the external test data: {len(test_dataset)}')

        test_loader = DataLoader(
            test_dataset,
            num_workers=cfg_dot.linear_probing_params.num_workers,
            batch_size=cfg_dot.linear_probing_params.batch_size,
            shuffle=False)

        test_params = {
            'test_loader': test_loader,
            'device': device,
            'model': model,
            'pretrained_cpt_dest': best_ckpt_destination,
            'metric_saving_path': f'./lp_evaluation_results/internal/{pth_base_name}_test_metrics_results.xlsx'
        }
        test_loop(test_params)

    elif dataset == 'vin':
        pass
    
    print('Finished probing evaluation without error :)')


def test_loop(params):

    test_loader = params['test_loader']
    device = params['device']
    model = params['model']
    metric_saving_path = params['metric_saving_path']
    delong_stats_saving_path = params['delong_stats_saving_path']
    
    all_labels = []
    all_preds = []
    all_probs = []

    # external:
    model.fc.load_state_dict(torch.load(params['pretrained_cpt_dest']))
    # internal 
    # model.load_state_dict(torch.load(params['pretrained_cpt_dest']))

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

    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1 Score', 'AUC', 'PR_AUC'], # the labels and the pred_probs are for delong auc significant test
        'Micro': [precision_micro, recall_micro, f1_micro, auc_micro, pr_auc_score_micro],
        'Weighted': [precision_weighted, recall_weighted, f1_weighted, auc_weighted, pr_auc_score_weighted],
        'Macro': [precision_macro, recall_macro, f1_macro, auc_macro, pr_auc_score_macro]
    }

    print('Saving the metrics results')
    # save the stats for delong computation
    assert(len(all_labels.flatten().tolist())==len(all_probs.flatten().tolist()))
    os.makedirs(os.path.dirname(delong_stats_saving_path), exist_ok=True)
    labels_preds = {
        'labels': all_labels.flatten().tolist(),
        'pred_probs': all_probs.flatten().tolist()
    }
    # Save to a pickle file
    with open(delong_stats_saving_path, "wb") as f:
        pickle.dump(labels_preds, f)
        print('finished dumping the files')

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


    
