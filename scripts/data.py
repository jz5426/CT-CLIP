import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm
import h5py
import math

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
    def __init__(self, 
                data_folder, 
                csv_file, 
                min_slices=20, 
                resize_dim=500, 
                force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.accession_to_text = None
        self.paths=[]
        self.accession_to_text = self.load_accession_text(csv_file)            
        self.samples = self.prepare_samples()
        print('number of files ', len(self.samples))

        self.count = 0
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
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

        data = {
            'ct': video_tensor,
            'report': input_text
        }
        return data
        # return video_tensor, input_text


class CustomCTDataset(Dataset):
    def __init__(self, 
                image_dir:str, 
                format:str='h5', 
                save_dir:str=None, 
                plot:bool=False):
        """
        Args:
            image_dir (str): Path to directory containing images.
            format (str): File formats of the images.
            save_dir (str): Directory to save the plots.
            plot (bool): Plot the sample images.
        """
        self.image_dir = image_dir
        self.format = format
        self.image_files = self.read_files(self.image_dir)
        self.save_dir = save_dir
        # self.plot = plot

    def read_files(self, directory):
        """
        Retrieve paths of all NIfTI files in the given directory.

        Args:
        directory (str): Path to the directory containing NIfTI files.

        Returns:
        list: List of paths to NIfTI files.
        """
        files = []
        for root, dirs, _files in os.walk(directory):
            for file in _files:
                if file.endswith(self.format.split('_')[-1]):
                    files.append(os.path.join(root, file))
        return files
    
    def __len__(self):
        return len(self.image_files)

class CustomCTReportDataset(CTReportDataset):
    def __init__(self, 
                 data_folder, 
                 csv_file, 
                 label_file=None,
                 split='train',
                 meta_data = None,
                 file_format='h5',
                 min_slices=20, 
                 resize_dim=500, 
                 force_num_frames=True
                ):
        assert split in ['train', 'val']
        if split == 'val':
            assert label_file is not None
            self.label_df = pd.read_csv(label_file)
            label_cols = list(self.label_df.columns[1:])
            self.label_df['one_hot_labels'] = list(self.label_df[label_cols].values)

        self.split = split
        self.format = file_format
        self.meta_data = meta_data # panda dataframe
        super().__init__(data_folder, csv_file, min_slices, resize_dim, force_num_frames)
        self.customCTDatasetObj = None

    def prepare_samples(self):
        self.customCTDatasetObj = CustomCTDataset(self.data_folder, format=self.format)
        samples = []
        for nii_file in self.customCTDatasetObj.image_files:
            accession_number = nii_file.split("/")[-1]
            accession_number = accession_number.replace(f".{self.format}", ".nii.gz")

            # make sure the vol has the corresponding reports
            if accession_number not in self.accession_to_text:
                continue

            impression_text = self.accession_to_text[accession_number]

            if impression_text == "Not given.":
                impression_text=""

            input_text_concat = ""
            for text in impression_text:
                input_text_concat = input_text_concat + str(text)
            input_text_concat = impression_text[0]

            obj = [nii_file, input_text_concat]
            if self.split == 'val':
                onehotlabels = self.label_df[self.label_df["VolumeName"] == accession_number]["one_hot_labels"].values
                # skip the ones without labels
                if len(onehotlabels) == 0:
                    continue
                obj.append(onehotlabels[0])

            samples.append(obj)
            self.paths.append(nii_file)

        return samples

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.split == 'val':
            nii_file, input_text, label = sample
        else:
            nii_file, input_text = sample
            
        # preprocess the custom compressed CT (.h5 file)
        video_tensor = self.preprocess_ct(nii_file)

        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')
        dir_path = os.path.splitext(nii_file)[0].split(os.sep)
        instance_name = dir_path[-1]

        data = {
            'ct': video_tensor,
            'report': input_text,
            'instance_name': instance_name,
            'ct_file_path':nii_file
        }
        if self.split == 'val':
            data['label'] = label
        return data

    def load(self, file):
        if self.format.endswith('pt'):
            return torch.load(file, weights_only=False)
        elif self.format.endswith('npz'):
            return torch.from_numpy(np.load(file)['arr_0'])
        elif self.format.endswith('h5'):
            with h5py.File(file, "r") as f:
                return f["ct"][:] # return only the numpy

    def preprocess_ct(self, nii_file):

        # preprocess the file in __get_item__
        img_data = self.load(nii_file)

        # check the ct image in f32 format
        # sitk.WriteImage(sitk.GetImageFromArray(img_data.astype(np.float32)), '/cluster/home/t135419uhn/CT-CLIP/ct_image/test.nii')
        
        file_name = os.path.basename(nii_file)
        row = self.meta_data[self.meta_data['VolumeName'] == file_name.replace(f".{self.format}", ".nii.gz")]
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])
        if math.isnan(z_spacing):
            z_spacing = xy_spacing

        # Define the target spacing values
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        def _scale_clip_resize(nii_data, current, target):

            # scale
            _img_data = slope * nii_data + intercept

            # clip
            hu_min, hu_max = -1000, 1000
            _img_data = np.clip(_img_data, hu_min, hu_max)
            _img_data = (((_img_data ) / 1000))#.astype(np.float32) # as float is important

            _img_data = _img_data.transpose(2, 0, 1) # becomes: z, x, y
            ct_tensor = torch.tensor(_img_data)
            ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)

            # resize
            _img_data = resize_array(ct_tensor, current, target)
            _img_data = _img_data[0][0]
            _img_data= np.transpose(_img_data, (1, 2, 0)) # xyz
            _img_data = _img_data*1000

            return _img_data

        current = (z_spacing, xy_spacing, xy_spacing)
        ct_image = _scale_clip_resize(
            img_data, 
            current, 
            (target_z_spacing, target_x_spacing, target_y_spacing)
            )
        
        # convert to float32 after preprocessing
        ct_image = ct_image.astype(np.float32)
        # check the CT images
        # sitk.WriteImage(sitk.GetImageFromArray(ct_image), '/cluster/home/t135419uhn/CT-CLIP/ct_image/f16_preprocessed.nii')

        # for ct
        tensor = torch.tensor(ct_image)
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

        
    def __len__(self):
        return len(self.samples)

