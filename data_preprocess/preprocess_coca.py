# /cluster/projects/mcintoshgroup/publicData/coca/cocacoronarycalciumandchestcts-2/Gated_release_final/patient

import os
import pydicom
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import pandas as pd
from pathlib import Path

def find_patients_subdirs(path):
    # based on the directory is named as number
    path = Path(path)
    return [str(p)+'/' for p in path.iterdir() if p.is_dir() and p.name.isdigit()]

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

# Function to convert DICOM files to NIfTI
def convert_dicom_to_cxr(patient_dcm_dir, output_dir):
    try:
        path_parts = Path(patient_dcm_dir).parts
        
        # Create output directory structure
        # example: '/cluster/projects/mcintoshgroup/publicData/coca/cocacoronarycalciumandchestcts-2/synthetic_xrays/deidentified_nongated/preprocessed_xray_mha/37'
        nifti_output_path = os.path.join(output_dir, 'preprocessed_xray_mha', path_parts[-1])
        rgb_output_path = os.path.join(output_dir, 'preprocessed_xray_rgb', path_parts[-1])
        if len(os.listdir(rgb_output_path)) > 0 and len(os.listdir(nifti_output_path)) > 0:
            return
        os.makedirs(nifti_output_path, exist_ok=True)
        os.makedirs(rgb_output_path, exist_ok=True)

        # Convert DICOM files to NIfTI format and save
        nifti_output_path = os.path.join(nifti_output_path, f'{path_parts[-1]}.mha')
        rgb_output_path = os.path.join(rgb_output_path, f'{path_parts[-1]}.rgb')

        # start real processing here.
        patient_dcm_dir_root = os.path.isdir(os.path.join(patient_dcm_dir, path_parts[-1])) if os.path.isdir(os.path.join(patient_dcm_dir, path_parts[-1])) else patient_dcm_dir
        dicom_files = [str(p) for p in Path(patient_dcm_dir_root).iterdir() if p.is_file() and p.suffix == '.dcm']
        if not dicom_files:
            return False

        dicom_slices = [pydicom.dcmread(f) for f in dicom_files]
        dicom_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        pixel_data = np.stack([s.pixel_array for s in dicom_slices])
        affine = np.eye(4)

        # Extract metadata: Rescale Slope, Rescale Intercept, and X, Y, Z Spacing
        slope = dicom_slices[0].RescaleSlope if hasattr(dicom_slices[0], 'RescaleSlope') else 1
        intercept = dicom_slices[0].RescaleIntercept if hasattr(dicom_slices[0], 'RescaleIntercept') else 0
        xy_spacing = dicom_slices[0].PixelSpacing if hasattr(dicom_slices[0], 'PixelSpacing') else [1, 1]
        x_spacing, y_spacing = xy_spacing[0], xy_spacing[1]
        slice_thickness = dicom_slices[0].SliceThickness if hasattr(dicom_slices[0], 'SliceThickness') else 1

        # manually find the slice spacing and the screening length of the CT
        z_positions = [ds.ImagePositionPatient[2] for ds in dicom_slices]

        # Calculate Z-spacing
        z_positions = [float(s.ImagePositionPatient[2]) for s in dicom_slices]
        z_spacing = np.abs(np.diff(z_positions).mean()) if len(z_positions) > 1 else slice_thickness

        nifti_image = nib.Nifti1Image(pixel_data, affine)
        img_data = nifti_image.get_fdata()

        # remove the defected vols (unreadable vols)
        #TODO: double check this one more time.
        if len(img_data.shape) != 3 or img_data.shape[0] == 1 or z_spacing == 0 or z_spacing == 0.0:
            print('skip processing files due to shape problem')
            return False

        #NOTE: rotate the axis so that it matches the ct orientation of the ct-rate dataset ORIGINAL shape: (65, 512, 512)
        img_data = np.rot90(img_data, k=-1, axes=(0,2)) 

        def _scale_clip_resize(nii_data, current, target):

            # scale
            _img_data = slope * nii_data + intercept

            # clip
            hu_min, hu_max = -1000, 1000
            _img_data = np.clip(_img_data, hu_min, hu_max)
            _img_data = (((_img_data ) / 1000)).astype(np.float32) # as float is important

            _img_data = _img_data.transpose(2, 0, 1) # becomes: z, x, y
            ct_tensor = torch.tensor(_img_data)
            ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)

            # resize
            _img_data = resize_array(ct_tensor, current, target)
            _img_data = _img_data[0][0]
            _img_data= np.transpose(_img_data, (1, 2, 0)) # xyz
            _img_data = _img_data*1000

            return _img_data

        current = (z_spacing, x_spacing, y_spacing)
        xray_image = _scale_clip_resize(img_data, current, (1,1,1))
        # for xray
        xray_image = sitk.GetImageFromArray(xray_image)
        mean_projection_filter = sitk.MeanProjectionImageFilter()
        mean_projection_filter.SetProjectionDimension(1)
        xray_image = mean_projection_filter.Execute(xray_image) # execute projection

        #NOTE: not sure why we need manual flipping here to match nii image for the frontal view
        xray_array = sitk.GetArrayFromImage(xray_image)
        xray_array = np.squeeze(xray_array) # make the image upright but NOTE that it is flipped with respect to the y-axis
        xray_array = np.flip(xray_array, axis=0)
        xray_array = np.rot90(xray_array, k=-1) # make the image upright but NOTE that it is flipped with respect to the y-axis

        #NOTE: make sure the xrays are upright
        np_image = (xray_array - xray_array.min()) / (xray_array.max() - xray_array.min()) * 255
        np_image = np_image.astype(np.uint8)  # Convert to uint8 for PIL compatibility
        rgb_image = np.stack([np_image] * 3, axis=-1)  # Shape: (H, W, 3)
        rgb_image = Image.fromarray(rgb_image, mode="RGB")
        rgb_image.save(rgb_output_path)
        
        xray_image = sitk.GetImageFromArray(xray_array)
        xray_image.SetSpacing((1.0, 1.0))  # Example spacing
        xray_image.SetOrigin((0.0, 0.0))   # Example origin
        sitk.WriteImage(xray_image, nifti_output_path)

        return True
    except Exception as e:
        print(f"Error processing {patient_dcm_dir}: {e}")
        return False


def main():

    # Directory paths
    ct_type = 'deidentified_nongated'
    input_dir = f'/cluster/projects/mcintoshgroup/publicData/coca/cocacoronarycalciumandchestcts-2/{ct_type}/'
    output_dir = f'/cluster/projects/mcintoshgroup/publicData/coca/cocacoronarycalciumandchestcts-2/synthetic_xrays/{ct_type}/'

    # Ensure output directory is created
    os.makedirs(output_dir, exist_ok=True)

    # Number of unique patients
    patient_dcm_dirs = find_patients_subdirs(input_dir)
    num_samples = len(patient_dcm_dirs)

    # Iterate over each sampled patient
    total_processed = 0
    progress_bar = tqdm(total=num_samples, desc="Processing Patients")

    for patient_dcm_dir in patient_dcm_dirs:
        results = convert_dicom_to_cxr(patient_dcm_dir, output_dir)
        if results:
            # update the progress only when a patient is successfully processed
            total_processed += 1
            progress_bar.update(1)

        if total_processed >= num_samples:
            break

if __name__ == "__main__":
    main() # NOTE: test bed without multiworkers
    
