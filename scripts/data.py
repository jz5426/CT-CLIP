import os
import glob
import json
from cxr_clip_utils import load_transform, transform_image
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm
import SimpleITK as sitk
import warnings
import random
from skmultilearn.model_selection import iterative_train_test_split

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

class CTReportDataset(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.accession_to_text = None
        self.paths=[]
        self.accession_to_text = self.load_accession_text(csv_file)            
        self.samples = self.prepare_samples()
        print('number of files ', len(self.samples))

        self.count = 0
        #self.resize_dim = resize_dim
        #self.resize_transform = transforms.Resize((resize_dim, resize_dim))
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            # accession_to_text[row['AccessionNo']] = row["Findings_EN"],row['Impressions_EN']
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text


    def prepare_samples(self):
        samples = []
        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))
                # nii_files = glob.glob(os.path.join(accession_folder, '*.pt'))
                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]
                    # accession_number = accession_number.replace(".pt", ".nii.gz")
                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]

                    if impression_text == "Not given.":
                        impression_text=""

                    input_text_concat = ""
                    for text in impression_text:
                        input_text_concat = input_text_concat + str(text)
                    input_text_concat = impression_text[0]
                    input_text = f'{impression_text}'
                    samples.append((nii_file, input_text_concat))
                    self.paths.append(nii_file)
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        df = pd.read_csv("train_metadata.csv") #select the metadata
        file_name = path.split("/")[-1]
        row = df[df['VolumeName'] == file_name]
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])

        # Define the target spacing values
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)

        img_data = slope * img_data + intercept
        img_data = img_data.transpose(2, 0, 1)

        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))

        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data ) / 1000)).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)

        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        return tensor


    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')

        return video_tensor, input_text

class CTReportDataSplitter:
    """mainly for the evaluation experiment"""
    def __init__(self, csv_file, labels, data_folder, xray_embeddings=None):
        self.labels = labels
        self.xray_paths = []
        self.data_folder = data_folder
        self.parent_folder = os.path.basename(data_folder)

        # optionally have the text and ct embeddings
        if xray_embeddings:
            self.xray_embeddings = torch.load(xray_embeddings)

        self.file_extension = 'mha' # make sure the xray data file path are the .mha file
        assert self.file_extension in data_folder
        self.accession_to_text = self.load_accession_text(csv_file)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            # accession_to_text[row['AccessionNo']] = row["Findings_EN"],row['Impressions_EN']
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text
    
    def prepare_samples(self, train_split=1., val_split=0.2):
        """
        this prepare the xray data in a dictionary format
        """

        # Read labels once outside the loop
        label_df = pd.read_csv(self.labels)
        test_label_cols = list(label_df.columns[1:])
        label_df['one_hot_labels'] = list(label_df[test_label_cols].values)

        samples = []
        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                mha_files = glob.glob(os.path.join(accession_folder, f'*.{self.file_extension}'))
                for xray_file in mha_files:
                    
                    # ignore the current .mha file if no corresponding rerport file
                    path_dirs = xray_file.split(os.sep)
                    accession_number = path_dirs[-1]
                    accession_number = accession_number.replace(f".{self.file_extension}", ".nii.gz")
                    if accession_number not in self.accession_to_text:
                        continue

                    # ignore the current .mha file if no label exists
                    onehotlabels = label_df[label_df["VolumeName"] == accession_number]["one_hot_labels"].values
                    if len(onehotlabels) == 0:
                        continue
                    
                    # TODO: get the corresponding xray embeddings
                    # instance_name = os.path.basename(xray_file)[:-len(f'.{self.file_extension}')]
                    samples.append((xray_file, onehotlabels[0]))

        # if the training size is smaller than 1, the internal validation should be extracted based on the splitted training set
        if train_split < 1.0:
            assert(val_split > 0)
            sample_data, sample_labels = [s[0] for s in samples], [s[-1] for s in samples]
            # First split to retain `train_split` amount of data
            train_data, train_label, _, _ = iterative_train_test_split(
                np.array(sample_data).reshape(-1, 1), 
                np.array(sample_labels), 
                test_size=(1.0 - train_split)
            )
            
            # Further split train_samples into train and validation using val_split
            train_data, train_label, val_data, val_label = iterative_train_test_split(
                np.array(train_data).reshape(-1, 1), 
                np.array(train_label), 
                test_size=val_split
            )
            
            train_split_samples = [(x[0], np.array(y)) for x, y in zip(train_data.tolist(), train_label.tolist())]
            val_split_samples = [(x[0], np.array(y)) for x, y in zip(val_data.tolist(), val_label.tolist())]

            print(f'training size: {len(train_split_samples)} validation size: {len(val_split_samples)}')
            return train_split_samples, val_split_samples

            # # If no val_split is desired, return the train samples
            # train_samples = [(x[0], np.array(y)) for x, y in zip(train_data.tolist(), train_label.tolist())]
            # return train_samples

        elif train_split == 1. and val_split > 0.:
            sample_data, sample_labels = [ s[0] for s in samples], [ s[-1] for s in samples]
            train_data, train_label, test_data, test_labels = iterative_train_test_split(np.array(sample_data).reshape(-1,1), np.array(sample_labels), test_size=val_split)
            val_split = [(x[0], np.array(y)) for x, y in zip(test_data.tolist(), test_labels.tolist())]
            train_split = [(x[0], np.array(y)) for x, y in zip(train_data.tolist(), train_label.tolist())]

            print(f'training size: {len(train_split)} validation size: {len(val_split)}')
            return train_split, val_split

        # return full results
        print(f'internal test size: {len(samples)}')
        return samples

class CTReportXRayClassificationDataset:

    def __init__(self,
                 cfg, 
                 data, # list of data processed from the prepare_sample
                 model_type,
                 data_embeddings=None,
                 split='train'):
        self.file_extension = 'mha'
        self.xray_paths = []
        self.cfg = cfg
        self.samples = data
        self.embeddings = data_embeddings
        self.normalize = 'huggingface' if 'swin' in model_type.lower() or 'vit' in model_type.lower() else 'imagenet' # when use swin or non-resnet architecture
        print('normalization used => ', self.normalize)

        # self.normalize = "huggingface" # when use swin or non-resnet architecture
        # if cfg["model"]["image_encoder"]["name"] == "resnet":
        #     self.normalize = "imagenet" # only for resnet architecture

        # TODO: check the split for validation set
        self.xray_transform = load_transform(split=split, transform_config=cfg['transform'])
            # image size 224, with clahe.yamel transformation during training and default.yaml transfomration during evaluation
            # if it is resnet, then use the imagenet normalization, otherwise use the huggingface normalization (.5).
        self.xray_to_rgb = partial(self.xray_mha_to_rgb, transform=self.xray_transform)

    def nii_img_to_tensor(self, path, transform):
        """
        override the parent method
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            img_data = torch.load(path)
        return img_data

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

    def __getitem__(self, key_id):

        selected_sample = self.samples[key_id] # based on index
        xray_file, label = selected_sample

        # get the corresonding embeddings
        name_acc = os.path.basename(xray_file)[:-len(f'.{self.file_extension}')]
        xray_embedding = self.embeddings[name_acc]

        # transformation borrowed from cxr_clip
        # xray_image = self.xray_to_rgb(xray_file)
        # xray_image = transform_image(self.xray_transform, xray_image, normalize=self.normalize)
        # return xray_image, label

        label = torch.from_numpy(label)
        return xray_embedding, label

    def __len__(self):
        return len(self.samples)

class MimicCTReportXRayDataset:
    """mainly used in retrieval evaluation and linear probe evaluation in the MimicCTClipInference class"""
    def __init__(self,
                 cfg, 
                 data_folder, # list of data processed from the prepare_sample
                 csv_file,
                 labels,
                 model_type,
                 split='valid'):
        self.file_extension = 'mha'
        self.xray_paths = []
        self.cfg = cfg
        self.labels = labels
        self.accession_to_text = self.load_accession_text(csv_file)
        self.samples = self.prepare_samples(data_folder)
        self.normalize = 'huggingface' if 'swin' in model_type.lower() or 'vit' in model_type.lower() else 'imagenet' # when use swin or non-resnet architecture
        print('normalization used => ', self.normalize)

        self.xray_transform = load_transform(split=split, transform_config=cfg['transform'])
            # image size 224, with clahe.yamel transformation during training and default.yaml transfomration during evaluation
            # if it is resnet, then use the imagenet normalization, otherwise use the huggingface normalization (.5).
        self.xray_to_rgb = partial(self.xray_mha_to_rgb, transform=self.xray_transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['hadm_id']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text

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
    
    def prepare_samples(self, data_folder):
        """
        this prepare the xray data in a dictionary format
        """
        # Read labels once outside the loop
        # NOTE: inside the label file, no extension is used for the hadm_id column
        label_df = pd.read_csv(self.labels)
        test_label_cols = list(label_df.columns[1:])
        label_df['one_hot_labels'] = list(label_df[test_label_cols].values)

        # based on the xray files and retrieve the corresponding report and labels
        samples = []
        for xray_file in tqdm.tqdm(glob.glob(os.path.join(data_folder, f'*.{self.file_extension}'))):
            # xray_file is the xray file path

            accession_number = os.path.basename(xray_file) # hadm_id.mha with extension
            accession_number = accession_number[:-len(f'.{self.file_extension}')] # hadm_id
            if accession_number not in self.accession_to_text:
                assert False

            impression_text = self.accession_to_text[accession_number] # finding + impresion
            text_final = ""
            for text in list(impression_text):
                text = str(text)
                if text == "Not given.":
                    text = ""
                text_final = text_final + ' ' + text

            # TODO: double check this.
            # get the filename without extension
            onehotlabels = label_df[label_df["hadm_id"] == accession_number]["one_hot_labels"].values
            if len(onehotlabels) == 0:
                assert False
            samples.append((xray_file, text_final, onehotlabels[0], accession_number))

        return samples


    def __getitem__(self, idx):

        selected_sample = self.samples[idx] # based on index
        xray_file, report, label, accession_number = selected_sample

        # check the CTReportDatasetinfer class
        report = report.replace('"', '')  
        report = report.replace('\'', '')  
        report = report.replace('(', '')  
        report = report.replace(')', '')
        # report = report.replace('_', '') # customly added

        # transformation borrowed from cxr_clip
        xray_image = self.xray_to_rgb(xray_file)
        xray_image = transform_image(self.xray_transform, xray_image, normalize=self.normalize)
        label = torch.from_numpy(label)
        return xray_image, report, label, accession_number # the instance_name

    def __len__(self):
        return len(self.samples)
    
# class VinBigDataChestXrayDataset(Dataset):
#     def __init__(self, dataframe, image_dir):
#         super().__init__()
#         self.image_ids = dataframe["image_id"].unique()
#         self.df = dataframe
#         self.image_dir = image_dir
#         self.transforms = 

#     def xray_mha_to_rgb(self, path, transform):
#         """
#         assume the path to the xray is mha format
#         """
        
#         # Step 1: Read the .mha file using SimpleITK
#         itk_image = sitk.ReadImage(path)
        
#         # Step 2: Convert to a NumPy array
#         np_image = sitk.GetArrayFromImage(itk_image)  # Shape: (H, W)

#         np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min()) * 255
#         np_image = np_image.astype(np.uint8)  # Convert to uint8 for PIL compatibility

#         rgb_image = np.stack([np_image] * 3, axis=-1)  # Shape: (H, W, 3)
#         rgb_image = Image.fromarray(rgb_image, mode="RGB")

#         return rgb_image

#     def __getitem__(self, index):

#         pass

class CTReportXRayDataset(CTReportDataset):

    def __init__(self,
                 data_folder, 
                 cfg, 
                 model_type,
                 csv_file='',
                 img_embedding_path='F:\\Chris\\dataset\\features_embeddings\\train\\image_features.pth', 
                 text_embedding_path='F:\\Chris\\dataset\\features_embeddings\\train\\text_features.pth', 
                 batch_style='patient', 
                 min_slices=20, 
                 resize_dim=500, 
                 force_num_frames=True):
        self.xray_paths = []
        self.parent_folder = os.path.basename(data_folder)
        assert(batch_style in ['patient', 'experiment', 'instance'])
        self.batch_style = batch_style
        self.ct_embeddings = torch.load(img_embedding_path)
        self.text_embeddings = torch.load(text_embedding_path)
        self.ct_embeddings = self._preprocess_embeddings(self.ct_embeddings, level=batch_style)
        self.text_embeddings = self._preprocess_embeddings(self.text_embeddings, level=batch_style)
        assert(self.ct_embeddings.keys() == self.text_embeddings.keys())
        self.file_extension = 'mha'

        super().__init__(data_folder, csv_file, min_slices, resize_dim, force_num_frames)
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

        self.normalize = 'huggingface' if 'swin' in model_type.lower() or 'vit' in model_type.lower() else 'imagenet' # when use swin or non-resnet architecture
        print('normalization used => ', self.normalize)

        # self.normalize = "huggingface" # when use swin or non-resnet architecture
        # if cfg["model"]["image_encoder"]["name"] == "resnet":
        #     self.normalize = "imagenet" # only for resnet architecture

        self.xray_transform = load_transform(split='train', transform_config=cfg['transform'])
            # image size 224, with clahe.yamel transformation during training and default.yaml transfomration during evaluation
            # if it is resnet, then use the imagenet normalization, otherwise use the huggingface normalization (.5).
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

            # group the latent embedding belong to the common identifier
            # example: if at the patient level: all reconstruction of all experiment for this patient are grouped.
            # example: if at the experiment level: all reconstruction for this experiment are grouped.
            if identifier not in processed_embeddings:
                processed_embeddings[identifier] = []
            processed_embeddings[identifier].append((embedding_dict[key], key))

        return processed_embeddings
    

    def nii_img_to_tensor(self, path, transform):
        """
        override the parent method
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            img_data = torch.load(path)
        return img_data

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

    def __getitem__(self, key_id):

        # Randomly select an image for this patient
        selected_sample = random.choice(self.samples[key_id])
        img_embedding, text_embedding, xray_file = selected_sample

        xray_image = self.xray_to_rgb(xray_file)
        # transformation borrowed from cxr_clip
        xray_image = transform_image(self.xray_transform, xray_image, normalize=self.normalize)

        img_embedding = torch.from_numpy(img_embedding.reshape(-1)).requires_grad_(False)
        text_embedding = torch.from_numpy(text_embedding.reshape(-1)).requires_grad_(False)

        name_acc = os.path.basename(xray_file)[:-len(f'.{self.file_extension}')]

        return  img_embedding, text_embedding, 'this is train', xray_image, name_acc, xray_file

    def prepare_samples(self):
        """
        this prepare the xray data in a dictionary format
        """
        # based on the xray files and retrieve the corresponding embeddings
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

                    # get ct and text embeddings
                    if patient not in samples:
                        samples[patient] = []
                    ct_embedding = next(filter(lambda x: x[1] == instance_name, self.ct_embeddings[patient]), None)[0]
                    text_embedding = next(filter(lambda x: x[1] == instance_name, self.text_embeddings[patient]), None)[0]
                    samples[patient].append((ct_embedding, text_embedding, xray_file))

        return samples

    def __len__(self):
        return len(self.key_ids)
