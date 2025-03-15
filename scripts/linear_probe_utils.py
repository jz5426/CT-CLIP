import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import CTReportDataSplitter, CTReportXRayClassificationDataset, MimicCTReportXRayDataset, RadChestXrayClassificationDataset, RadChestXrayDataset, RadChestXraySplitter, VinBigChestXrayClassificationDataset, VinBigChestXrayDataSplitter, VinBigDataChestXrayDataset
from eval_utils import XrayClassificationModel, get_clean_model_name, metadata_base_on_model_type
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
import shutil
import constants as const

def load_cached_ct_rate_xray_features(pth_base_name, split):
    # split = 'train'

    # base on the baseline model, load the corresponding xray features
    xray_feature_path = f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/xray_features_embeddings/{split}/{pth_base_name}'
    xray_features = torch.load(xray_feature_path)
    print('Xray feature extraction completed')
    return xray_features

def get_train_internal_split(cfg_dot, cfg):
    """implementation copied from internal_split_caching.py"""

    # get the metadata
    _, xray_model_type, pth_base_name, _ = metadata_base_on_model_type(
        cfg_dot.linear_probing_params.baseline_type,
        pth_trailing_string='features')

    if cfg_dot.linear_probing_params.evaluation_dataset == const.NLST:
        print('Splitting NLST dataset')
        
        # TODO:

        # base on the baseline model, load the corresponding xray features
        xray_feature_path = f'/cluster/projects/mcintoshgroup/publicData/NLST/xray_features_embeddings/train/{pth_base_name}'
        train_xray_features = torch.load(xray_feature_path)

        pass
    elif cfg_dot.linear_probing_params.evaluation_dataset == 'mimic':
        print('Splitting ct-rate mimic version dataset: differences in the set of the labels')

        # base on the baseline model, load the corresponding xray features
        xray_feature_path = f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/xray_features_embeddings/train/{pth_base_name}'
        train_xray_features = torch.load(xray_feature_path)

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

        train_dataset = CTReportXRayClassificationDataset(
            cfg=cfg,
            data=train_sample, # actual data potentially with the embeddings
            data_embeddings=train_xray_features,
            model_type=xray_model_type,
            split='train'
        )

        internal_val_dataset = CTReportXRayClassificationDataset(
            cfg=cfg,
            data=internal_val_samples, # actual data potentially with the embeddings
            data_embeddings=train_xray_features,
            model_type=xray_model_type,
            split='train'
        )
        results = {
            'train_dataset': train_dataset,
            'internal_val_dataset': internal_val_dataset
        }
    elif cfg_dot.linear_probing_params.evaluation_dataset == 'ct-rate':
        print('Splitting ct-rate dataset')

        # base on the baseline model, load the corresponding xray features
        xray_feature_path = f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/xray_features_embeddings/train/{pth_base_name}'
        train_xray_features = torch.load(xray_feature_path)

        # Set up the dataset and data loaders
        #NOTE: the label is the mimic version (with 11 labels) but the report and the data are the original CT-RATE
        train_data_splitter = CTReportDataSplitter(
            csv_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
            labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv', #NOTE: the label need to be the ct-rate version
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
            model_type=xray_model_type,
            split='train'
        )

        internal_val_dataset = CTReportXRayClassificationDataset(
            cfg=cfg,
            data=internal_val_samples, # actual data potentially with the embeddings
            data_embeddings=train_xray_features,
            model_type=xray_model_type,
            split='train'
        )

        results = {
            'train_dataset': train_dataset,
            'internal_val_dataset': internal_val_dataset
        }
    elif cfg_dot.linear_probing_params.evaluation_dataset in [const.RADCHEST_CT, const.RADCHEST_CT_PURE]:

        print('Splitting ct-rate rachest_ct version dataset: differences in the set of the labels')
        # base on the baseline model, load the corresponding xray features
        xray_feature_path = f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/xray_features_embeddings/train/{pth_base_name}'
        train_xray_features = torch.load(xray_feature_path)

        # Set up the dataset and data loaders
        #NOTE: the label is the radchest_ct version (with 15 labels) but the report and the data are the original CT-RATE
            # particularly, the calcification related labels are merged.
        dataset = cfg_dot.linear_probing_params.evaluation_dataset 
        if dataset == const.RADCHEST_CT:
            labels = '/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_radchest_ct_labels.csv' 
        elif dataset == const.RADCHEST_CT_PURE:
            labels = '/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_radchest_ct_pure_labels.csv'
        train_data_splitter = CTReportDataSplitter(
            csv_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/train_reports.csv',
            labels=labels, #NOTE: the label need to be the radchest_ct or radchest_ct_pure version
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
            model_type=xray_model_type,
            split='train'
        )

        internal_val_dataset = CTReportXRayClassificationDataset(
            cfg=cfg,
            data=internal_val_samples, # actual data potentially with the embeddings
            data_embeddings=train_xray_features,
            model_type=xray_model_type,
            split='train'
        )

        results = {
            'train_dataset': train_dataset,
            'internal_val_dataset': internal_val_dataset
        }
    elif cfg_dot.linear_probing_params.evaluation_dataset in [const.RADCHEST_CT_PURE_INTERNAL, const.RADCHEST_CT_INTERNAL, const.RADCHEST_CT_PURE_INTERNAL_CLEAN, const.RADCHEST_CT_INTERNAL_CLEAN, const.RADCHEST_CT_ALL_DISEASE_INTERNAL, const.RADCHEST_CT_ALL_DISEASE_CT_ONLY_INTERNAL]:
        print(f'Splitting {cfg_dot.linear_probing_params.evaluation_dataset} dataset')
    
        dataset = cfg_dot.linear_probing_params.evaluation_dataset
        # base on the baseline model, load the corresponding xray features
        xray_feature_path = f'/cluster/projects/mcintoshgroup/publicData/RADChestCT/{cfg_dot.linear_probing_params.evaluation_dataset}/xray_features_embeddings/valid/{pth_base_name}'
        train_xray_features = torch.load(xray_feature_path)
        dataset = cfg_dot.linear_probing_params.evaluation_dataset

        # what kind of specific radchest ct data. 
        if dataset == const.RADCHEST_CT_INTERNAL: # without empty multi-hot vector maximum overlap with CT-RATE in the labels but potentially have xray labels
            labels = '/cluster/projects/mcintoshgroup/publicData/RADChestCT/final_labels_clean.csv' 
        elif dataset == const.RADCHEST_CT_PURE_INTERNAL: 
            labels = '/cluster/projects/mcintoshgroup/publicData/RADChestCT/final_labels_pure_clean.csv' 
        elif dataset == const.RADCHEST_CT_PURE_INTERNAL_CLEAN: # without empty multi-hot vector but for few-shot adaption
            labels = '/cluster/projects/mcintoshgroup/publicData/RADChestCT/final_labels_pure_clean.csv' 
        elif dataset == const.RADCHEST_CT_INTERNAL_CLEAN: # without empty multi-hot vector maximum overlap with CT-RATE in the labels but potentially have xray labels
            labels = '/cluster/projects/mcintoshgroup/publicData/RADChestCT/final_labels_clean.csv' 
        elif dataset == const.RADCHEST_CT_ALL_DISEASE_INTERNAL: # with xray labels, not fair for few-shot adaptation 
            labels = '/cluster/projects/mcintoshgroup/publicData/RADChestCT/final_labels_all_CT_disease.csv'
        elif dataset == const.RADCHEST_CT_ALL_DISEASE_CT_ONLY_INTERNAL: # for few-shot adaptation task
            labels = '/cluster/projects/mcintoshgroup/publicData/RADChestCT/final_labels_CT_only_disease.csv'

        data_splitter = RadChestXraySplitter(
            labels=labels,
            data_folder='/cluster/projects/mcintoshgroup/publicData/RADChestCT/preprocessed_xray_mha'
        )
        #NOTE: note that train_data_portion should be higher as it only contains 3630 images
        train_sample, internal_val_samples, test_samples = data_splitter.prepare_samples(
            train_split=cfg_dot.linear_probing_params.train_data_portion,
            val_split=0.2
        ) # validation split is always, train_split is controlable

        train_dataset = RadChestXrayClassificationDataset(
            cfg=cfg,
            data=train_sample, # actual data potentially with the embeddings
            data_embeddings=train_xray_features,
            model_type=xray_model_type,
            split='train'
        )

        internal_val_dataset = RadChestXrayClassificationDataset(
            cfg=cfg,
            data=internal_val_samples, # actual data potentially with the embeddings
            data_embeddings=train_xray_features,
            model_type=xray_model_type,
            split='valid'
        )

        # do the same thing for test samples but make sure the labels are correct.
        test_dataset = RadChestXrayClassificationDataset(
            cfg=cfg,
            data=test_samples, # actual data potentially with the embeddings
            data_embeddings=train_xray_features,
            model_type=xray_model_type,
            split='valid'
        )

        results = {
            'train_dataset': train_dataset,
            'internal_val_dataset': internal_val_dataset,
            'test_dataset': test_dataset
        }
    elif 'vinBig' in cfg_dot.linear_probing_params.evaluation_dataset: # the ct dataset
        print(f'Splitting {cfg_dot.linear_probing_params.evaluation_dataset} dataset')
    
        dataset = cfg_dot.linear_probing_params.evaluation_dataset
        split = 'train'
        # base on the baseline model, load the corresponding xray features
        xray_feature_path = f'/cluster/projects/mcintoshgroup/publicData/VinBigDataChestXray/{dataset}/xray_features_embeddings/train/{pth_base_name}'
        train_xray_features = torch.load(xray_feature_path)

        train_data_splitter = VinBigChestXrayDataSplitter(
            labels=f'/cluster/projects/mcintoshgroup/publicData/VinBigDataChestXray/image_labels_{split}.csv', #NOTE: the label need to be the mha version
            data_folder=f'/cluster/projects/mcintoshgroup/publicData/VinBigDataChestXray/preprocessed_vinbig_{split}/vinbig_preprocessed_xray_mha',
            label_variant=cfg_dot.linear_probing_params.evaluation_dataset
        )
        train_sample, internal_val_samples = train_data_splitter.prepare_samples(
            train_split=cfg_dot.linear_probing_params.train_data_portion,
            val_split=0.2
        ) # validation split is always, train_split is controlable


        train_dataset = VinBigChestXrayClassificationDataset(
            cfg=cfg,
            data=train_sample, # actual data potentially with the embeddings
            data_embeddings=train_xray_features,
            model_type=xray_model_type,
            split=split
        )

        internal_val_dataset = VinBigChestXrayClassificationDataset(
            cfg=cfg,
            data=internal_val_samples, # actual data potentially with the embeddings
            data_embeddings=train_xray_features,
            model_type=xray_model_type,
            split=split
        )
    
        results = {
            'train_dataset': train_dataset,
            'internal_val_dataset': internal_val_dataset,
        }

    return results

def get_pathologies(dataset='ct-rate'):
    # NOTE: the order of the listed pathologies matter
    if dataset == 'ct-rate':
      pathologies = ['Medical material',
                  'Arterial wall calcification', #calcification
                  'Cardiomegaly', #
                  'Pericardial effusion', #
                  'Coronary artery wall calcification', #calcification
                  'Hiatal hernia', # hernia
                  'Lymphadenopathy', #
                  'Emphysema', #
                  'Atelectasis', #
                  'Lung nodule', #nodule
                  'Lung opacity', #opacity
                  'Pulmonary fibrotic sequela', # fibrosis
                  'Pleural effusion', #
                  'Mosaic attenuation pattern',
                  'Peribronchial thickening', #bronchial wall thickening or periboncail ** double check this: bronchial_wall_thickening, ‘pleural_thickening, pericardial thickening
                  'Consolidation', #
                  'Bronchiectasis', #
                  'Interlobular septal thickening'] # septal thickening
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
        pathologies = [
            'Aortic enlargement',
            'Atelectasis',
            'Calcification',
            'Cardiomegaly',
            'Clavicle fracture',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Enlarged PA',
            'ILD',
            'Infiltration',
            'Lung Opacity',
            'Lung cavity',
            'Lung cyst',
            'Mediastinal shift',
            'Nodule/Mass',
            'Pleural effusion',
            'Pleural thickening',
            'Pneumothorax',
            'Pulmonary fibrosis',
            'Rib fracture',
            'COPD',
            'Lung tumor',
            'Pneumonia',
            'Tuberculosis'
        ]
    elif dataset == 'vinBig_ct':
        pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Emphysema', 'Lung Opacity', 'Pleural effusion']
    elif dataset in [const.RADCHEST_CT_INTERNAL, const.RADCHEST_CT_INTERNAL_CLEAN, const.RADCHEST_CT]:
        # _INTERNAL means uses the radchest ct data to train the classifier and evaluate on the remaining
        pathologies = [
            'calcification',
            'Cardiomegaly',
            'pericardial_effusion',
            'hernia',
            'Lymphadenopathy',
            'Emphysema',
            'Atelectasis',
            'nodule',
            'opacity',
            'fibrosis',
            'pleural_effusion',
            'bronchial_wall_thickening', # assumed
            'Consolidation',
            'Bronchiectasis',
            'septal_thickening'
        ]
        pathologies = [p.lower() for p in pathologies]
    elif dataset in [const.RADCHEST_CT_PURE_INTERNAL,const.RADCHEST_CT_PURE_INTERNAL_CLEAN, const.RADCHEST_CT_PURE]:
        pathologies = [
            'calcification',
            'pericardial_effusion',
            'hernia',
            'lymphadenopathy',
            'emphysema',
            'fibrosis',
            'bronchial_wall_thickening',
            'bronchiectasis',
            'septal_thickening'
        ]
    elif dataset == const.RADCHEST_CT_ALL_DISEASE_INTERNAL:
        pathologies = [
            "tree_in_bud",
            "air_trapping",
            "bronchiolectasis",
            "bronchiolitis",
            "cyst",
            "honeycombing",
            "groundglass",
            "septal_thickening",
            "mucous_plugging",
            "pleural_thickening",
            "pericardial_thickening",
            "coronary_artery_disease",
            "aneurysm",
            "atherosclerosis",
            "granuloma",
            "nodulegr1cm",
            "opacity",
            "plaque",
            "scattered_nod"
        ]
    elif dataset == const.RADCHEST_CT_ALL_DISEASE_CT_ONLY_INTERNAL:
        pathologies = [
            "tree_in_bud",
            "bronchiolectasis",
            "bronchiolitis",
            "groundglass",
            "septal_thickening",
            "pericardial_thickening",
            "coronary_artery_disease",
            "aneurysm",
            "atherosclerosis"
        ]
    elif dataset == const.NLST:
        pathologies = [
            'cvd',
            'no_cvd'
        ]
    return pathologies


def linear_probing_main(params):
    model = params['model']
    device = params['device']
    train_dataset = params['train_dataset']
    internal_val_dataset = params['internal_val_dataset']
    cfg_dot = params['cfg_dot']
    ckpt_parent_dir = params['ckpt_parent_dir']
    best_ckpt_destination = params['best_ckpt_destination']

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

    print("Finetuning the Xray encoder completed ==> perform external testing")
    return model

def evaluate_classifier(params):
    dataset = params['dataset']
    cfg = params['cfg']
    cfg_dot = params['cfg_dot']
    clip_xray = params['clip_xray']
    device = params['device']
    xray_model_type = params['xray_model_type']
    model = params['model']
    best_ckpt_destination = params['best_ckpt_destination']
    pth_base_name = params['pth_base_name']

    test_params = {
        'device': device,
        'dataset': dataset,
        'xray_model_type': xray_model_type,
        'train_data_portion': cfg_dot.linear_probing_params.train_data_portion,
        'cfg_dot': cfg_dot
    }

    if dataset == 'mimic':
        test_dataset = MimicCTReportXRayDataset(
            cfg=cfg,
            data_folder='/cluster/home/t135419uhn/CT-CLIP/preprocessed_mimic/mimic_preprocessed_xray_mha',
            csv_file='/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/external_valid_mimic_report.csv',
            labels='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_external_valid_mimic_labels.csv', 
            model_type=xray_model_type,
            split='valid'
        )
        print(f'size of the external test data: {len(test_dataset)}')
        
        test_loader = DataLoader(
            test_dataset, 
            num_workers=cfg_dot.linear_probing_params.num_workers, 
            batch_size=cfg_dot.linear_probing_params.test_loader_batch_size, 
            shuffle=False)

        # define a full classifier with encoder and projection layer learnt from the CT-RATE training set
        # NOTE: since mimic is a new dataset, so it requires full forward pass of the vision encoder and then the projection layer
        classification_model = XrayClassificationModel(
            vision_model=clip_xray.xray_encoder, 
            feature_projector=clip_xray.to_xray_latent, 
            pretrained_classifier=model, # load the classifier layer, the pretrained weight will be loaded soon
            vision_model_type=xray_model_type
        )
        classification_model.to(device)

        test_params = {
            **test_params,
            'test_loader': test_loader,
            'model': classification_model,
            'full_forward_pass': True,
            'pretrained_cpt_dest': best_ckpt_destination, # where to retrieve the best checkpoint in the test loop
        }
        return test_loop(test_params)
    elif dataset == 'ct-rate':

        # split=valid is the test set for internal validation and the pth_base_name is mainly use to retrieve the xray features of the particular backbone.
        val_xray_features = load_cached_ct_rate_xray_features(pth_base_name, split='valid') # the split to be tested.
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
            data_embeddings=val_xray_features, # the xray embeddings of a particular backbone.
            model_type=xray_model_type,
            split='valid'
        )
        print(f'size of the external test data: {len(test_dataset)}')

        #NOTE: we do not need full forward pass as we already saved the preprocessed validation features of the xray
        test_loader = DataLoader(
            test_dataset,
            num_workers=cfg_dot.linear_probing_params.num_workers,
            batch_size=cfg_dot.linear_probing_params.test_loader_batch_size,
            shuffle=False)

        test_params = {
            **test_params,
            'test_loader': test_loader,
            'model': model,
            'full_forward_pass': False,
            'pretrained_cpt_dest': best_ckpt_destination, # destination to retreive the checkpoint for the linear classifier only.
        }
        return test_loop(test_params)
    elif dataset in [const.RADCHEST_CT, const.RADCHEST_CT_PURE]:
        if dataset == const.RADCHEST_CT:
            labels = '/cluster/projects/mcintoshgroup/publicData/RADChestCT/final_labels_clean.csv'
        elif dataset == const.RADCHEST_CT_PURE:
            # this file should be created in preprocess_radchestct_labels.py
            labels = '/cluster/projects/mcintoshgroup/publicData/RADChestCT/final_labels_pure_clean.csv'
        test_dataset = RadChestXrayDataset(
            data_folder = '/cluster/projects/mcintoshgroup/publicData/RADChestCT/preprocessed_xray_mha',
            model_type=xray_model_type,
            cfg=cfg,
            labels=labels
        )
        print(f'size of the external radchest_ct data: {len(test_dataset)}')

        test_loader = DataLoader(
            test_dataset, 
            num_workers=cfg_dot.linear_probing_params.num_workers, 
            batch_size=cfg_dot.linear_probing_params.test_loader_batch_size, 
            shuffle=False)

        classification_model = XrayClassificationModel(
            vision_model=clip_xray.xray_encoder, # from pretrained
            feature_projector=clip_xray.to_xray_latent, # from pretrained
            pretrained_classifier=model, # load the classifier layer, the pretrained weight will be loaded in test_loop function
            vision_model_type=xray_model_type
        )
        classification_model.to(device)

        test_params = {
            **test_params,
            'test_loader': test_loader,
            'model': classification_model,
            'full_forward_pass': True,
            'pretrained_cpt_dest': best_ckpt_destination, # where to retrieve the best checkpoint
        }
        return test_loop(test_params)
    elif dataset in [const.RADCHEST_CT_INTERNAL, const.RADCHEST_CT_PURE_INTERNAL, const.RADCHEST_CT_INTERNAL_CLEAN, const.RADCHEST_CT_PURE_INTERNAL_CLEAN, const.RADCHEST_CT_ALL_DISEASE_INTERNAL, const.RADCHEST_CT_ALL_DISEASE_CT_ONLY_INTERNAL]:
        test_dataset = params['test_data']
        print(f'size of the external radchest_ct data: {len(test_dataset)}')

        test_loader = DataLoader(
            test_dataset, 
            num_workers=cfg_dot.linear_probing_params.num_workers, 
            batch_size=cfg_dot.linear_probing_params.test_loader_batch_size, 
            shuffle=False)

        test_params = {
            **test_params,
            'test_loader': test_loader,
            'model': model,
            'full_forward_pass': False, # if ran xray_feature_caching with this dataset => False, otherwise True
            'pretrained_cpt_dest': best_ckpt_destination, # where to retrieve the best checkpoint
        }
        return test_loop(test_params)

    elif 'vinBig' in dataset:
        #NOTE: follow similarly to the mimic external validaion.
        split = 'test'
        test_dataset = VinBigDataChestXrayDataset(
            cfg=cfg,
            data_folder=f'/cluster/projects/mcintoshgroup/publicData/VinBigDataChestXray/preprocessed_vinbig_{split}/vinbig_preprocessed_xray_mha',
            labels=f'/cluster/projects/mcintoshgroup/publicData/VinBigDataChestXray/image_labels_{split}.csv', 
            model_type=xray_model_type,
            label_variant=dataset,
            split=split)

        # Split dataset into train and validation sets
        test_loader = DataLoader(
            test_dataset,
            num_workers=cfg_dot.linear_probing_params.num_workers,
            batch_size=cfg_dot.linear_probing_params.batch_size,
            shuffle=False)
    
        classification_model = XrayClassificationModel(
            vision_model=clip_xray.xray_encoder, # from pretrained
            feature_projector=clip_xray.to_xray_latent, # from pretrained
            pretrained_classifier=model, # load the classifier layer, the pretrained weight will be loaded in test_loop function
            vision_model_type=xray_model_type
        )
        classification_model.to(device)

        test_params = {
            **test_params,
            'test_loader': test_loader,
            'model': classification_model,
            'full_forward_pass': True,
            'pretrained_cpt_dest': best_ckpt_destination, # where to retrieve the best checkpoint
        }
        return test_loop(test_params)
    elif dataset == const.NLST:
        #TODO:
        pass

    print('something wrong')


def test_loop(params):

    test_loader = params['test_loader']
    device = params['device']
    model = params['model']
    full_forward_pass = params['full_forward_pass']
    dataset = params['dataset']
    xray_model_type = params['xray_model_type']
    train_portion = params['train_data_portion']
    cfg = params['cfg_dot']
    auc_type = cfg.linear_probing_params.auc_type
    
    all_labels = []
    all_preds = []
    all_probs = []

    # for full forward pass, only need to load the weights of the classifier
    if full_forward_pass:
        model.fc.load_state_dict(torch.load(params['pretrained_cpt_dest']))
    else:
        # default to be the classifier layer
        model.load_state_dict(torch.load(params['pretrained_cpt_dest']))

    print(f'Performing testing with size (in unit batch) {len(test_loader)}')
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            # inputs, _, labels, _ = data
            inputs, labels = data['xray'], data['label']
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
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=auc_type)
    except Exception as e:
        precision, recall, f1 = -1, -1, -1

    try:
        auc = roc_auc_score(all_labels, all_probs, average=auc_type, multi_class='ovr')
    except Exception as e:
        auc = -1
    
    try:
        pr_auc_score = average_precision_score(all_labels, all_probs, average=auc_type)
    except Exception as e:
        pr_auc_score = -1

    # compute aucroc for each class in the multihot vector
    auc_per_class = []
    for i in range(all_labels.shape[1]):
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except Exception as e:
            auc = -1
        auc_per_class.append(auc.item() if isinstance(auc, np.float64) else auc)

    print(f"Test Results for {auc_type} average: F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, AUC: {auc:.4f}, PR_AUC: {pr_auc_score:.4f}")

    assert(len(all_labels.flatten().tolist())==len(all_probs.flatten().tolist()))

    # each row has the following results
    metric_results = {
        const.DATASET: [dataset], # evaluation dataset
        const.MODEL: [get_clean_model_name(xray_model_type)], # the xray model that the metrics belong to 
        const.FEW_SHOT: [train_portion], # the few-shots
        const.AUC: [auc],
        const.PR_AUC: [pr_auc_score],
        const.LABELS: [all_labels.flatten().tolist()],
        const.PRED_PROBS: [all_probs.flatten().tolist()],
        const.PER_CLASS_AUC: [auc_per_class]
    }

    return metric_results


def validation_loop(params):
    val_loader = params['val_loader']
    device = params['device']
    model = params['model']
    criterion = params['criterion']

    model.eval()
    val_loss = 0.0
    print(f'Performing validation with size (in unit batch) {len(val_loader)}')
    with torch.no_grad():
        for data in val_loader:
            # inputs, _, labels, _ = data
            inputs, labels = data['xray'], data['label']
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
        # inputs, _, labels, _ 
        inputs, labels = data['xray'], data['label']
        
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


    
