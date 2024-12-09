import os
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image
import math
from functools import partial
import shutil

# "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/metadata/dataset_metadata_validation_metadata.csv"
df = pd.read_csv('C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\dataset\\metadata\\train_metadata.csv') #select the metadata
# df = pd.read_csv('/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/metadata/train_metadata.csv')


def read_nii_files(directory):
    """
    Retrieve paths of all NIfTI files in the given directory.

    Args:
    directory (str): Path to the directory containing NIfTI files.

    Returns:
    list: List of paths to NIfTI files.
    """
    nii_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))
    return nii_files

def read_nii_data(file_path):
    """
    Read NIfTI file data.

    Args:
    file_path (str): Path to the NIfTI file.

    Returns:
    np.ndarray: NIfTI file data.
    """
    try:
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        return nii_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

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

def process_file(file_path, split, shared_dst_dir): #'F:\\Chris\\dataset'
    """
    Process a single NIfTI file.

    Args:
    file_path (str): Path to the NIfTI file.
    shared_dst_dir: parent path to store the files
    Returns:
    None
    """
    original_file_name = os.path.basename(file_path)
    file_name = os.path.basename(file_path)
    # should check if the file exists before preceed the loading so that save computation resources
    ct_save_folder = "{}_preprocessed_ct".format(split) #save folder for preprocessed
    ct_folder_path_new = os.path.join(shared_dst_dir, ct_save_folder, "{}_".format(split) + file_name.split("_")[1], "{}_".format(split) + file_name.split("_")[1] + file_name.split("_")[2]) #folder name for train or validation
    os.makedirs(ct_folder_path_new, exist_ok=True)
    file_name = file_name.split(".")[0]+".pt"
    ct_save_path = os.path.join(ct_folder_path_new, file_name)

    xray_save_folder = "{}_preprocessed_xray_mha".format(split) #save folder for preprocessed
    xray_folder_path_new = os.path.join(shared_dst_dir, xray_save_folder, "{}_".format(split) + file_name.split("_")[1], "{}_".format(split) + file_name.split("_")[1] + file_name.split("_")[2]) #folder name for train or validation
    os.makedirs(xray_folder_path_new, exist_ok=True)
    file_name = file_name.split(".")[0]+".mha"
    xray_save_path = os.path.join(xray_folder_path_new, file_name)

    xray_rgb_save_folder = "{}_preprocessed_xray_rgb".format(split) #save folder for preprocessed
    xray_folder_path_new = os.path.join(shared_dst_dir, xray_rgb_save_folder, "{}_".format(split) + file_name.split("_")[1], "{}_".format(split) + file_name.split("_")[1] + file_name.split("_")[2]) #folder name for train or validation
    os.makedirs(xray_folder_path_new, exist_ok=True)
    file_name = file_name.split(".")[0]+".png"
    xray_rgb_save_path = os.path.join(xray_folder_path_new, file_name)

    try:
        img_data = read_nii_data(file_path)
        if img_data is None:
            print(f"Read {file_path} unsuccessful. Passing")
            return

        row = df[df['VolumeName'] == original_file_name]
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
            _img_data = (((_img_data ) / 1000)).astype(np.float32) # as float is important

            _img_data = _img_data.transpose(2, 0, 1) # z, x, y
            ct_tensor = torch.tensor(_img_data)
            ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)

            # resize
            _img_data = resize_array(ct_tensor, current, target)
            _img_data = _img_data[0][0]
            _img_data= np.transpose(_img_data, (1, 2, 0)) # xyz
            _img_data = _img_data*1000

            return _img_data

        current = (z_spacing, xy_spacing, xy_spacing)

        ct_image = _scale_clip_resize(img_data, current, (target_z_spacing, target_x_spacing, target_y_spacing))
        xray_image = _scale_clip_resize(img_data, current, (1,1,1))

        #TEST
        # sitk.WriteImage(sitk.GetImageFromArray(ct_image), './test_{}'.format(original_file_name))
        
        # for xray
        xray_image = sitk.GetImageFromArray(xray_image)
        mean_projection_filter = sitk.MeanProjectionImageFilter()
        mean_projection_filter.SetProjectionDimension(1)
        xray_image = mean_projection_filter.Execute(xray_image) # execute projection

        #NOTE: not sure why we need manual flipping here to match nii image for the frontal view
        xray_array = sitk.GetArrayFromImage(xray_image)
        # xray_array = np.flip(np.squeeze(xray_array), axis=0)
        xray_array = np.rot90(np.squeeze(xray_array)) # make the image upright but NOTE that it is flipped with respect to the y-axis
        # xray_array = np.squeeze(xray_array) # make the image upright but NOTE that it is flipped with respect to the y-axis

        np_image = (xray_array - xray_array.min()) / (xray_array.max() - xray_array.min()) * 255
        np_image = np_image.astype(np.uint8)  # Convert to uint8 for PIL compatibility
        rgb_image = np.stack([np_image] * 3, axis=-1)  # Shape: (H, W, 3)
        rgb_image = Image.fromarray(rgb_image, mode="RGB")
        # rgb_image.show()

        xray_image = sitk.GetImageFromArray(xray_array)
        xray_image.SetSpacing((1.0, 1.0))  # Example spacing
        xray_image.SetOrigin((0.0, 0.0))   # Example origin

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

        # save the ct image as a pt tensor
        torch.save(tensor, ct_save_path) # save as .pt file # np.savez(save_path, resized_array)
        if os.path.exists(file_path):
            os.remove(file_path)  # Remove the file

        # save the xray as a .mha image
        sitk.WriteImage(xray_image, xray_save_path)

        # save the xray image as .png image
        rgb_image.save(xray_rgb_save_path)
        
        return
    except:
        with open('./defect_ct.txt', 'a') as file:
            file.write(f'{original_file_name}\n')
        return

    # original_file_name = os.path.basename(file_path)
    # file_name = os.path.basename(file_path)
    # # should check if the file exists before preceed the loading so that save computation resources
    # ct_save_folder = "train_preprocessed_ct" #save folder for preprocessed
    # ct_folder_path_new = os.path.join(shared_dst_dir, ct_save_folder, "train_" + file_name.split("_")[1], "train_" + file_name.split("_")[1] + file_name.split("_")[2]) #folder name for train or validation
    # os.makedirs(ct_folder_path_new, exist_ok=True)
    # file_name = file_name.split(".")[0]+".pt"
    # ct_save_path = os.path.join(ct_folder_path_new, file_name)

    # xray_save_folder = "train_preprocessed_xray_mha" #save folder for preprocessed
    # xray_folder_path_new = os.path.join(shared_dst_dir, xray_save_folder, "train_" + file_name.split("_")[1], "train_" + file_name.split("_")[1] + file_name.split("_")[2]) #folder name for train or validation
    # os.makedirs(xray_folder_path_new, exist_ok=True)
    # file_name = file_name.split(".")[0]+".mha"
    # xray_save_path = os.path.join(xray_folder_path_new, file_name)

    # xray_rgb_save_folder = "train_preprocessed_xray_rgb" #save folder for preprocessed
    # xray_folder_path_new = os.path.join(shared_dst_dir, xray_rgb_save_folder, "train_" + file_name.split("_")[1], "train_" + file_name.split("_")[1] + file_name.split("_")[2]) #folder name for train or validation
    # os.makedirs(xray_folder_path_new, exist_ok=True)
    # file_name = file_name.split(".")[0]+".png"
    # xray_rgb_save_path = os.path.join(xray_folder_path_new, file_name)

    # #NOTE: check and proceed
    # # if os.path.exists(ct_save_path) and os.path.exists(xray_save_path) and os.path.exists(xray_rgb_save_path):
    # #     return

    # # proceed with unsaved data

    # img_data = read_nii_data(file_path)
    # if img_data is None:
    #     print(f"Read {file_path} unsuccessful. Passing")
    #     return

    # row = df[df['VolumeName'] == original_file_name]
    # slope = float(row["RescaleSlope"].iloc[0])
    # intercept = float(row["RescaleIntercept"].iloc[0])
    # xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
    # z_spacing = float(row["ZSpacing"].iloc[0])
    # if math.isnan(z_spacing):
    #     z_spacing = xy_spacing
    # # Define the target spacing values
    # target_x_spacing = 0.75
    # target_y_spacing = 0.75
    # target_z_spacing = 1.5

    # def _scale_clip_resize(nii_data, current, target):

    #     # scale
    #     _img_data = slope * nii_data + intercept

    #     # clip
    #     hu_min, hu_max = -1000, 1000
    #     _img_data = np.clip(_img_data, hu_min, hu_max)
    #     _img_data = (((_img_data ) / 1000)).astype(np.float32) # as float is important

    #     _img_data = _img_data.transpose(2, 0, 1) # z, x, y
    #     ct_tensor = torch.tensor(_img_data)
    #     ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)

    #     # resize
    #     _img_data = resize_array(ct_tensor, current, target)
    #     _img_data = _img_data[0][0]
    #     _img_data= np.transpose(_img_data, (1, 2, 0)) # xyz
    #     _img_data = _img_data*1000

    #     return _img_data

    # current = (z_spacing, xy_spacing, xy_spacing)

    # ct_image = _scale_clip_resize(img_data, current, (target_z_spacing, target_x_spacing, target_y_spacing))
    # xray_image = _scale_clip_resize(img_data, current, (1,1,1))

    # #TEST
    # # sitk.WriteImage(sitk.GetImageFromArray(ct_image), './test_{}'.format(original_file_name))
    
    # # for xray
    # xray_image = sitk.GetImageFromArray(xray_image)
    # mean_projection_filter = sitk.MeanProjectionImageFilter()
    # mean_projection_filter.SetProjectionDimension(1)
    # xray_image = mean_projection_filter.Execute(xray_image) # execute projection

    # xray_array = sitk.GetArrayFromImage(xray_image)
    # # xray_array = np.flip(np.squeeze(xray_array), axis=0)
    # xray_array = np.rot90(np.squeeze(xray_array)) # make the image upright but NOTE that it is flipped with respect to the y-axis
    # # xray_array = np.squeeze(xray_array) # make the image upright but NOTE that it is flipped with respect to the y-axis

    # np_image = (xray_array - xray_array.min()) / (xray_array.max() - xray_array.min()) * 255
    # np_image = np_image.astype(np.uint8)  # Convert to uint8 for PIL compatibility
    # rgb_image = np.stack([np_image] * 3, axis=-1)  # Shape: (H, W, 3)
    # rgb_image = Image.fromarray(rgb_image, mode="RGB")
    # # rgb_image.show()

    # xray_image = sitk.GetImageFromArray(xray_array)
    # xray_image.SetSpacing((1.0, 1.0))  # Example spacing
    # xray_image.SetOrigin((0.0, 0.0))   # Example origin

    # # for ct
    # tensor = torch.tensor(ct_image)
    # # Get the dimensions of the input tensor
    # target_shape = (480,480,240)

    # # Extract dimensions
    # h, w, d = tensor.shape

    # # Calculate cropping/padding values for height, width, and depth
    # dh, dw, dd = target_shape
    # h_start = max((h - dh) // 2, 0)
    # h_end = min(h_start + dh, h)
    # w_start = max((w - dw) // 2, 0)
    # w_end = min(w_start + dw, w)
    # d_start = max((d - dd) // 2, 0)
    # d_end = min(d_start + dd, d)

    # # Crop or pad the tensor
    # tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    # pad_h_before = (dh - tensor.size(0)) // 2
    # pad_h_after = dh - tensor.size(0) - pad_h_before

    # pad_w_before = (dw - tensor.size(1)) // 2
    # pad_w_after = dw - tensor.size(1) - pad_w_before

    # pad_d_before = (dd - tensor.size(2)) // 2
    # pad_d_after = dd - tensor.size(2) - pad_d_before

    # tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)
    # tensor = tensor.permute(2, 0, 1)
    # tensor = tensor.unsqueeze(0)

    # # save the ct image as a pt tensor
    # torch.save(tensor, ct_save_path) # save as .pt file # np.savez(save_path, resized_array)
    
    # # save the xray as a .mha image
    # sitk.WriteImage(xray_image, xray_save_path)

    # # save the xray image as .png image
    # rgb_image.save(xray_rgb_save_path)

# Example usage:
if __name__ == "__main__":
    #NOTE: TRY IT ON WINNDOWS
    split = 'train' # change the split to valid to process the validation data instead.

    # split_to_preprocess = '/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/valid' #select the validation or test split
    raw_ct_dir = f"F:\\Chris\\CT-RATE-temp\\dataset\\{split}" #select the validation or test split
    # split_to_preprocess = '/mnt/f/Chris/CT-RATE-temp/dataset/train' #select the validation or test split

    nii_files = read_nii_files(raw_ct_dir)
    num_workers = 8  # Number of worker processes

    # df = pd.read_csv('/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/metadata/train_metadata.csv')

    # Process files using multiprocessing with tqdm progress bar
    # F:\\Chris\\dataset\\CT-RATE-temp\\processed_dataset
    # '/mnt/f/Chris/CT-RATE-temp/processed_dataset'
    with Pool(num_workers) as pool:
        func_with_arg = partial(process_file, split=split, shared_dst_dir='F:\\Chris\\CT-RATE-temp\\processed_dataset')
        list(tqdm(pool.imap_unordered(func_with_arg, nii_files), total=len(nii_files)))

    print('    removing raw ct files\n')
    shutil.rmtree(raw_ct_dir)