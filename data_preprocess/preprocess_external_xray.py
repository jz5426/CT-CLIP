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

XRAY_DATA_TYPE = 'vinbigxray'
MIRROR = False
SPLIT = 'test'

def read_mimic_dcm_files(csv_file_path):
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

def read_vinbig_dcm_files(csv_file_path):
    df = pd.read_csv(csv_file_path)
    image_ids = df['image_id'].unique().tolist() # note that there are duplicates for multi labels
    print(f'unique: {len(image_ids)}')
    # create path base on each image id in the list
    vinbig_path_list = [(image_id, os.path.join(os.path.dirname(csv_file_path), SPLIT, image_id+'.dicom')) for image_id in image_ids]
    return vinbig_path_list

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
    image_id, file_path = file_instance_tuple
    if XRAY_DATA_TYPE == 'mimic':
        xray_folder_path_new = os.path.join(shared_dst_dir, 'mimic_preprocessed_xray_mha' if not MIRROR else 'mimic_preprocessed_xray_mha_mirror')
    elif XRAY_DATA_TYPE == 'vinbigxray':
        xray_folder_path_new = os.path.join(shared_dst_dir, 'vinbig_preprocessed_xray_mha' if not MIRROR else 'vinbig_preprocessed_xray_mha_mirror')

    os.makedirs(xray_folder_path_new, exist_ok=True)
    file_name = image_id + '.mha'
    xray_mha_save_path = os.path.join(xray_folder_path_new, file_name)

    if XRAY_DATA_TYPE == 'mimic':
        xray_folder_path_new = os.path.join(shared_dst_dir, 'mimic_preprocessed_xray_rgb' if not MIRROR else 'mimic_preprocessed_xray_rgb_mirror')
    elif XRAY_DATA_TYPE == 'vinbigxray':
        xray_folder_path_new = os.path.join(shared_dst_dir, 'vinbig_preprocessed_xray_rgb' if not MIRROR else 'vinbig_preprocessed_xray_rgb_mirror')

    os.makedirs(xray_folder_path_new, exist_ok=True)
    file_name = image_id + '.png'
    xray_rgb_save_path = os.path.join(xray_folder_path_new, file_name)

    xray_array = read_dcm_data(file_path)
    if xray_array is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return
    
    if 'mirror' in xray_folder_path_new: # rotate the image?
        print('mirroring the xray')
        xray_array = np.flip(np.squeeze(xray_array), axis=-1)

    # NOTE that this is only done to the rgb image, which is not saved.
    np_image = (xray_array - xray_array.min()) / (xray_array.max() - xray_array.min()) * 255
    np_image = np_image.astype(np.uint8)  # Convert to uint8 for PIL compatibility

    rgb_image = np.stack([np_image] * 3, axis=-1)  # Shape: (H, W, 3)
    rgb_image = Image.fromarray(rgb_image, mode="RGB")
    # rgb_image.show()

    xray_image = sitk.GetImageFromArray(xray_array)
    xray_image.SetSpacing((1.0, 1.0))  # Example spacing
    xray_image.SetOrigin((0.0, 0.0))   # Example origin

    # save the xray as a .mha image
    sitk.WriteImage(xray_image, xray_mha_save_path)

    # save the xray image as .png image
    rgb_image.save(xray_rgb_save_path)


if __name__ == "__main__":
    num_workers = 20
    if XRAY_DATA_TYPE == 'mimic':
        csv_file_path = '/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/mimic_ct_report_paired_with_ordered_label_pa_ap.csv'
        shared_dst_dir = './preprocessed_mimic'
        dcm_files = read_mimic_dcm_files(csv_file_path)
    elif XRAY_DATA_TYPE == 'vinbigxray':
        csv_file_path = f'/cluster/projects/mcintoshgroup/publicData/VinBigDataChestXray/image_labels_{SPLIT}.csv'
        shared_dst_dir = f'/cluster/projects/mcintoshgroup/publicData/VinBigDataChestXray/preprocessed_vinbig_{SPLIT}'
        dcm_files = read_vinbig_dcm_files(csv_file_path)

    with Pool(num_workers) as pool:
        func_with_arg = partial(process_file, shared_dst_dir=shared_dst_dir)
        list(tqdm(pool.imap_unordered(func_with_arg, dcm_files), total=len(dcm_files)))
