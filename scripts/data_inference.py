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


class CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True, labels = "labels.csv", probing_mode=False, load_assession=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.accession_to_text = None
        if load_assession:
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

    #NOTE: the following is the draft code for testing
    # def nii_img_to_tensor(self, path, transform): # this implementation is pre-process + the original nii_img_to_tensor
    #     nii_img = nib.load(str(path))
    #     img_data = nii_img.get_fdata()

    #     df = pd.read_csv("/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/metadata/dataset_metadata_validation_metadata.csv") #select the metadata
    #     file_name = path.split("/")[-1]
    #     row = df[df['VolumeName'] == file_name]
    #     slope = float(row["RescaleSlope"].iloc[0])
    #     intercept = float(row["RescaleIntercept"].iloc[0])
    #     xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
    #     z_spacing = float(row["ZSpacing"].iloc[0])

    #     # Define the target spacing values
    #     target_x_spacing = 0.75
    #     target_y_spacing = 0.75
    #     target_z_spacing = 1.5

    #     current = (z_spacing, xy_spacing, xy_spacing)
    #     target = (target_z_spacing, target_x_spacing, target_y_spacing)

    #     # scale for both xray and ct
    #     img_data = slope * img_data + intercept
    #     img_data_for_xray = img_data.copy() # deep copy

    #     # clip
    #     hu_min, hu_max = -1000, 1000
    #     img_data = np.clip(img_data, hu_min, hu_max)
    #     img_data = (((img_data ) / 1000)).astype(np.float32) # as float is important

    #     # clip xray intensity
    #     img_data_for_xray = np.clip(img_data_for_xray, hu_min, hu_max)
    #     img_data_for_xray = (((img_data_for_xray ) / 1000)).astype(np.float32) # as float is important

    #     img_data = img_data.transpose(2, 0, 1) # z, x, y
    #     tensor = torch.tensor(img_data)
    #     tensor = tensor.unsqueeze(0).unsqueeze(0)

    #     # resize for ct
    #     img_data = resize_array(tensor, current, target)
    #     img_data = img_data[0][0]
    #     img_data= np.transpose(img_data, (1, 2, 0)) # xyz
    #     img_data = img_data*1000

    #     #NOTE: save this image to check the orientation
    #     # ct_image = sitk.GetImageFromArray(img_data)
    #     # saving_path = os.path.join('./', '{}.nii'.format(file_name[:-len('.nii.gz')]))
    #     # sitk.WriteImage(ct_image, saving_path)

    #     # resize xray
    #     img_data_for_xray = resize_array(tensor, current, (1,1,1))
    #     img_data_for_xray = img_data_for_xray[0][0]
    #     img_data_for_xray= np.transpose(img_data_for_xray, (1, 2, 0)) # xyz
    #     img_data_for_xray = img_data_for_xray*1000

    #     ct_image = sitk.GetImageFromArray(img_data_for_xray)
    #     mean_projection_filter = sitk.MeanProjectionImageFilter()
    #     mean_projection_filter.SetProjectionDimension(1)
    #     xray_image = mean_projection_filter.Execute(ct_image)
    #     xray_array = sitk.GetArrayFromImage(xray_image)
    #     xray_array = np.flip(np.squeeze(xray_array), axis=0)
        
    #     xray_image = sitk.GetImageFromArray(xray_array)
    #     xray_image.SetSpacing((1.0, 1.0))  # Example spacing
    #     xray_image.SetOrigin((0.0, 0.0))   # Example origin
    #     saving_path = os.path.join('./', '{}.mha'.format(file_name[:-len('.nii.gz')]))
    #     sitk.WriteImage(xray_image, saving_path)

    #     plt.imshow(xray_array, cmap="gray")
    #     plt.tight_layout()
    #     plt.axis("off")
    #     plt.savefig('./test_{}.png'.format(file_name[:-len('.nii.gz')]), bbox_inches='tight')

    #     tensor = torch.tensor(img_data)
    #     # Get the dimensions of the input tensor
    #     target_shape = (480,480,240)

    #     # Extract dimensions
    #     h, w, d = tensor.shape

    #     # Calculate cropping/padding values for height, width, and depth
    #     dh, dw, dd = target_shape
    #     h_start = max((h - dh) // 2, 0)
    #     h_end = min(h_start + dh, h)
    #     w_start = max((w - dw) // 2, 0)
    #     w_end = min(w_start + dw, w)
    #     d_start = max((d - dd) // 2, 0)
    #     d_end = min(d_start + dd, d)

    #     # Crop or pad the tensor
    #     tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    #     pad_h_before = (dh - tensor.size(0)) // 2
    #     pad_h_after = dh - tensor.size(0) - pad_h_before

    #     pad_w_before = (dw - tensor.size(1)) // 2
    #     pad_w_after = dw - tensor.size(1) - pad_w_before

    #     pad_d_before = (dd - tensor.size(2)) // 2
    #     pad_d_after = dd - tensor.size(2) - pad_d_before

    #     tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

    #     tensor = tensor.permute(2, 0, 1)

    #     tensor = tensor.unsqueeze(0)

    #     return tensor

    #NOTE: the following is the original implementation
    # def nii_img_to_tensor(self, path, transform):

    #     nii = nib.load(path)
    #     img_data = nii.get_fdata()
    #     # img_data = np.load(path)['arr_0']
    #     # img_data= np.transpose(img_data, (1, 2, 0)) #NOTE: previously uncommented
    #     img_data = img_data*1000

    #     #NOTE: preprocessing the intensity values before crop and pad
    #     hu_min, hu_max = -1000, 1000
    #     img_data = np.clip(img_data, hu_min, hu_max)
    #     img_data = (((img_data+400 ) / 600)).astype(np.float32)
    #     slices=[]

    #     tensor = torch.tensor(img_data)
    #     # Get the dimensions of the input tensor
    #     target_shape = (480,480,240)
    #     # Extract dimensions
    #     h, w, d = tensor.shape

    #     # Calculate cropping/padding values for height, width, and depth, NOTE: this is center cropping
    #     dh, dw, dd = target_shape
    #     h_start = max((h - dh) // 2, 0)
    #     h_end = min(h_start + dh, h)
    #     w_start = max((w - dw) // 2, 0)
    #     w_end = min(w_start + dw, w)
    #     d_start = max((d - dd) // 2, 0)
    #     d_end = min(d_start + dd, d)

    #     # Crop or pad the tensor
    #     tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    #     pad_h_before = (dh - tensor.size(0)) // 2
    #     pad_h_after = dh - tensor.size(0) - pad_h_before

    #     pad_w_before = (dw - tensor.size(1)) // 2
    #     pad_w_after = dw - tensor.size(1) - pad_w_before

    #     pad_d_before = (dd - tensor.size(2)) // 2
    #     pad_d_after = dd - tensor.size(2) - pad_d_before

    #     tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

    #     #NOTE: transform the tensor back to the original shape
    #     tensor = tensor.permute(2, 0, 1) # depth as the channel size: [Batch Size, Channels, Height, Width]
    #     tensor = tensor.unsqueeze(0)
    #     return tensor

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

    def __init__(self, data_folder, cfg, min_slices=20, resize_dim=500, force_num_frames=True, labels="labels.csv", probing_mode=False):
        self.xray_paths = []
        super().__init__(data_folder, '', min_slices, resize_dim, force_num_frames, labels, probing_mode, load_assession=True)
        self.cfg = cfg

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

        self.normalize = "huggingface" # when use swin or non-resnet architecture
        if cfg["model"]["image_encoder"]["name"] == "resnet":
            self.normalize = "imagenet" # only for resnet architecture

        self.xray_transform = load_transform(split='valid', transform_config=cfg['transform'])
            # image size 224, with clahe.yamel transformation during training and default.yaml transfomration during evaluation
            # if it is resnet, then use the imagenet normalization, otherwise use the huggingface normalization (.5).
            # NOTE: inference transformation would be different than that during training.
        self.xray_to_rgb = partial(self.xray_mha_to_rgb, transform=self.xray_transform)

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

        # # Step 3: Use torch.from_numpy for fast conversion (shares memory)
        # tensor_image = torch.from_numpy(np_image)
        
        # # Step 4: Ensure the tensor has the correct dtype
        # tensor_image = tensor_image.to(torch.float32)
        
        # # Step 5: Normalize NOTE: should be according to the cxr_clip, the inference version
        # # tensor_image = (tensor_image - tensor_image.min()) / (tensor_image.max() - tensor_image.min())
        
        # # Step 6: Add channel dimension for PyTorch (C x 3 x H x W)
        # tensor_image = tensor_image.unsqueeze(0)  # Add batch dimension
        
        return rgb_image


    def prepare_samples(self):
        samples = []
        xray_path_dirs = self.xray_data_folder.split(os.sep)
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
                    path_dirs = nii_file.split(os.sep)
                    accession_number = path_dirs[-1]

                    accession_number = accession_number.replace(".pt", ".nii.gz")

                    # corresponding xray file
                    xray_file = os.sep.join(xray_path_dirs + path_dirs[path_dirs.index('valid_preprocessed_ct')+1:])
                    xray_file = xray_file.replace('.pt', '.mha')

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
                        samples.append((nii_file, text_final, onehotlabels[0], xray_file))
                        self.paths.append(nii_file)
                        self.xray_paths.append(xray_file)
        return samples


    def __getitem__(self, index):
        nii_file, input_text, onehotlabels, xray_file = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file) if not self.probing_mode else ['untoggle this']

        xray_image = self.xray_to_rgb(xray_file)
        xray_image = transform_image(self.xray_transform, xray_image, normalize=self.normalize)

        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')  
        name_acc = nii_file.split(os.sep)[-2]
        return video_tensor, input_text, onehotlabels, xray_image, name_acc, nii_file # add the nii_file for xray projections
