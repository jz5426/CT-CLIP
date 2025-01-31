import os
import glob
import json
from cxr_clip_utils import load_transform, transform_image
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import tqdm
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import warnings
from data import resize_array
import random

class CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True, labels = "labels.csv", probing_mode=False):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)
        self.probing_mode = probing_mode

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                # nii_files = glob.glob(os.path.join(accession_folder, '*.npz'))
                # nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz')) #NOTE: self modified
                nii_files = glob.glob(os.path.join(accession_folder, '*.pt'))


                for nii_file in nii_files:
                    # accession_number = nii_file.split("/")[-1]
                    accession_number = nii_file.split(os.sep)[-1]

                    # accession_number = accession_number.replace(".npz", ".nii.gz")
                    accession_number = accession_number.replace(".pt", ".nii.gz")
                    # accession_number = accession_number.replace(".nii", ".nii.gz") #NOTE: self modified
                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]
                    text_final = ""
                    for text in list(impression_text):
                        text = str(text)
                        if text == "Not given.":
                            text = ""

                        text_final = text_final + text

                    onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
                    if len(onehotlabels) > 0:
                        samples.append((nii_file, text_final, onehotlabels[0]))
                        self.paths.append(nii_file)
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform): 
        """
        path: this should be the path for the processed data instead of the original data
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            img_data = torch.load(path)

        return img_data

    def __getitem__(self, index):
        nii_file, input_text, onehotlabels = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file) if not self.probing_mode else ['untoggle this']
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')
        dir_path = os.path.splitext(nii_file)[0].split(os.sep)
        name_acc = dir_path[-2] 
        instance_name = dir_path[-1]
        return video_tensor, input_text, onehotlabels, name_acc, instance_name, nii_file # add the nii_file for xray projections

class CTReportXRayDatasetinfer(CTReportDatasetinfer):

    def __init__(self,
                 data_folder, 
                 model_type,
                 cfg, 
                 csv_file='',
                 img_embedding_path='F:\\Chris\\dataset\\features_embeddings\\valid\\image_features.pth', 
                 text_embedding_path='F:\\Chris\\dataset\\features_embeddings\\valid\\text_features.pth', 
                 batch_style='patient', 
                 min_slices=20, 
                 resize_dim=500, 
                 force_num_frames=True, 
                 labels="labels.csv", 
                 probing_mode=False):
        self.xray_paths = []
        assert(batch_style in ['patient', 'experiment', 'instance'])
        self.batch_style = batch_style
        self.ct_embeddings = torch.load(img_embedding_path)
        self.text_embeddings = torch.load(text_embedding_path)
        self.ct_embeddings = self._preprocess_embeddings(self.ct_embeddings, level=batch_style)
        self.text_embeddings = self._preprocess_embeddings(self.text_embeddings, level=batch_style)
        assert(self.ct_embeddings.keys() == self.text_embeddings.keys())
        self.file_extension = 'mha' # file extension for the xray files

        super().__init__(data_folder, csv_file, min_slices, resize_dim, force_num_frames, labels, probing_mode)
        self.cfg = cfg
        self.key_ids = list(self.samples.keys())
        # from trainer.py in cxr_clip
        # the following is not needed for now as we use the synthic paired xray
        # data_config = {}
        # if "data_train" in cfg:
        #     data_config["train"] = cfg["data_train"]
        # if "data_valid" in cfg:
        #     data_config["valid"] = cfg["data_valid"]
        # if "data_test" in cfg:
        #     data_config["test"] = cfg["data_test"]
        # if cfg["model"]["image_encoder"]["name"] == "resnet":
        #     for _split in data_config:
        #         for _dataset in data_config[_split]:
        #             data_config[_split][_dataset]["normalize"] = "imagenet"

        self.normalize = 'huggingface' if 'swin' in model_type or 'vit' in model_type else 'imagenet' # when use swin or non-resnet architecture
        # self.normalize = "huggingface" # when use swin or non-resnet architecture
        # if cfg["model"]["image_encoder"]["name"] == "resnet":
        #     self.normalize = "imagenet" # only for resnet architecture

        self.xray_transform = load_transform(split='valid', transform_config=cfg['transform'])
            # image size 224, with clahe.yamel transformation during training and default.yaml transfomration during evaluation
            # if it is resnet, then use the imagenet normalization, otherwise use the huggingface normalization (.5).
            # NOTE: inference transformation would be different than that during training.
        self.xray_to_rgb = partial(self.xray_mha_to_rgb, transform=self.xray_transform)


    def _preprocess_embeddings(self, embedding_dict, level='patient'):
        keys = list(embedding_dict.keys())
        processed_embeddings = {}

        for key in keys:
            key_parts = key.split('_')

            if level == 'patient':
                identifier = '_'.join(key_parts[:2])
            elif level == 'experiment':
                identifier = '_'.join(key_parts[:3])
            elif level == 'instance':
                identifier = key

            if identifier not in processed_embeddings:
                processed_embeddings[identifier] = []
            processed_embeddings[identifier].append((embedding_dict[key], key))

        return processed_embeddings

    def __getitem__(self, key_id):

        # Randomly select an random instance for this id (patient, experiment, instance)
        selected_sample = random.choice(self.samples[key_id])
        img_embedding, text_embedding, onehotlabels, xray_file = selected_sample

        xray_image = self.xray_to_rgb(xray_file)
        # transformation borrowed from cxr_clip
        xray_image = transform_image(self.xray_transform, xray_image, normalize=self.normalize)

        # name_acc = xray_file.split(os.sep)[-2] #TODO: double check this, this is being used in ctclip_feature_extraction function in run_zero_shot.py
        name_acc = os.path.basename(xray_file)[:-len(f'.{self.file_extension}')]

        img_embedding = torch.from_numpy(img_embedding.reshape(-1)).requires_grad_(False)
        text_embedding = torch.from_numpy(text_embedding.reshape(-1)).requires_grad_(False)
        return  img_embedding, text_embedding, onehotlabels, xray_image, name_acc, xray_file

    
    def xray_mha_to_rgb(self, path, transform):
        """
        assume the path to the xray is mha format
        """
        # Step 1: Read the .mha file using SimpleITK
        itk_image = sitk.ReadImage(path)
        
        # Step 2: Convert to a NumPy array
        np_image = sitk.GetArrayFromImage(itk_image)  # Shape: (H, W)

        np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min()) * 255
        np_image = np_image.astype(np.uint8)  # Convert to uint8 for PIL compatibility

        rgb_image = np.stack([np_image] * 3, axis=-1)  # Shape: (H, W, 3)
        rgb_image = Image.fromarray(rgb_image, mode="RGB")

        return rgb_image

    def prepare_samples(self):
        """
        this prepare the xray data in a dictionary format
        """
        # based on the xray files and retrieve the corresponding embeddings

        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        samples = {}
        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                mha_files = glob.glob(os.path.join(accession_folder, f'*.{self.file_extension}'))
                for xray_file in mha_files:
                    # get the filename without extension
                    instance_name = os.path.basename(xray_file)[:-len(f'.{self.file_extension}')]
                    key_parts = instance_name.split('_')

                    if self.batch_style == 'patient':
                        patient = '_'.join(key_parts[:2]) # patient level
                    elif self.batch_style == 'experiment':
                        patient = '_'.join(key_parts[:3]) # patient experiment level
                    elif self.batch_style == 'instance':
                        patient = instance_name # instance level (the original implementation)
                    
                    #TODO: double check this.
                    path_dirs = xray_file.split(os.sep)
                    accession_number = path_dirs[-1]
                    accession_number = accession_number.replace(f".{self.file_extension}", ".nii.gz")
                    if accession_number not in self.accession_to_text:
                        continue
                    # ignore the current .mha file if no label exists
                    onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
                    if len(onehotlabels) == 0:
                        continue

                    # get ct and text embeddings
                    if patient not in samples:
                        samples[patient] = []
                    ct_embedding = next(filter(lambda x: x[1] == instance_name, self.ct_embeddings[patient]), None)[0]
                    text_embedding = next(filter(lambda x: x[1] == instance_name, self.text_embeddings[patient]), None)[0]
                    samples[patient].append(
                        (ct_embedding, text_embedding, onehotlabels[0], xray_file)
                    )

        return samples