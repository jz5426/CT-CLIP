"""
should follow similar steps as in preprocess_ctrate_with_xray_train.py

mainly this is for mimic xray data
"""

import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image
from functools import partial
import pydicom

def read_dcm_files(csv_file_path):
    """
    Retrieve paths of all dcm files in the given directory.

    Returns:
    list: List of paths to dcm files.
    """

    df = pd.read_csv(csv_file_path)
    # Retrieve the 'cxr_path' column as a list
    cxr_path_list = df['cxr_path'].tolist()
    hadm_ids = df['hadm_id'].tolist()

    # h mainly used for naming the xray vols
    return [(h,c) for h, c in zip(hadm_ids, cxr_path_list)]

def read_dcm_data(file_path):
    """
    Read NIfTI file data.

    Args:
    file_path (str): Path to the NIfTI file.

    Returns:
    np.ndarray: NIfTI file data.
    """
    try:
        dicom = pydicom.dcmread(file_path)
        pixel_array = dicom.pixel_array
        return pixel_array
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def process_file(file_instance_tuple, shared_dst_dir):
    hadm_id, file_path = file_instance_tuple

    xray_folder_path_new = os.path.join(shared_dst_dir, 'mimic_preprocessed_xray_mha')
    os.makedirs(xray_folder_path_new, exist_ok=True)
    file_name = hadm_id + '.mha'
    xray_mha_save_path = os.path.join(xray_folder_path_new, file_name)

    xray_folder_path_new = os.path.join(shared_dst_dir, 'mimic_preprocessed_xray_rgb')
    os.makedirs(xray_folder_path_new, exist_ok=True)
    file_name = hadm_id + '.png'
    xray_rgb_save_path = os.path.join(xray_folder_path_new, file_name)

    xray_array = read_dcm_data(file_path)
    if xray_array is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return
    
    if 'mirror' in xray_folder_path_new: # rotate the image?
        print('mirroring the xray')
        xray_array = np.flip(np.squeeze(xray_array), axis=-1)
    np_image = (xray_array - xray_array.min()) / (xray_array.max() - xray_array.min()) * 255
    np_image = np_image.astype(np.uint8)  # Convert to uint8 for PIL compatibility

    rgb_image = np.stack([np_image] * 3, axis=-1)  # Shape: (H, W, 3)
    rgb_image = Image.fromarray(rgb_image, mode="RGB")
    rgb_image.show()

    xray_image = sitk.GetImageFromArray(xray_array)
    xray_image.SetSpacing((1.0, 1.0))  # Example spacing
    xray_image.SetOrigin((0.0, 0.0))   # Example origin

    # save the xray as a .mha image
    sitk.WriteImage(xray_image, xray_mha_save_path)

    # save the xray image as .png image
    rgb_image.save(xray_rgb_save_path)


if __name__ == "__main__":
    num_workers = 20
    csv_file_path = '/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/mimic_ct_report_paired_with_ordered_label_pa_ap.csv'
    dcm_files = read_dcm_files(csv_file_path)

    with Pool(num_workers) as pool:
        func_with_arg = partial(process_file, shared_dst_dir='./preprocessed_mimic')
        list(tqdm(pool.imap_unordered(func_with_arg, dcm_files), total=len(dcm_files)))
