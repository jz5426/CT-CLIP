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

def filter_split(
        input_dir='/mnt/g/NLST/manifest-NLST_allCT/',
        nlst_split_csv='/mnt/g/NLST/manifest-NLST_allCT/NLST_data_split_from_CVD_risk_estimator.csv', 
        nlst_metadata='/mnt/g/NLST/manifest-NLST_allCT/metadata.csv',
        split='ALL'
    ):

    # Read CSV files
    filtered_df = pd.read_csv(nlst_split_csv)
    metadata_df = pd.read_csv(nlst_metadata)
    
    # Filter rows where 'group' column equals 'TEST'
    if split != 'ALL':
        filtered_df = filtered_df[filtered_df['group'] == split]
    
    # Merge with metadata based on 'vol' matching 'Series UID'
    merged_df = filtered_df.merge(metadata_df[['Series UID', 'File Location']],
                                  left_on='vol', right_on='Series UID', how='inner')
    
    # Convert Windows-style paths to Unix-style paths
    merged_df['File Location'] = merged_df['File Location'].str.replace('\\', '/')
    # Drop 'Series UID' column as it's redundant after merging
    merged_df.drop(columns=['Series UID'], inplace=True)

    # Sanity check: Print warning if any rows have missing 'File Location'
    missing_file_location = merged_df['File Location'].isna().sum()
    if missing_file_location > 0:
        print(f"Warning: {missing_file_location} rows have missing 'File Location' values.")

    # Check if DICOM files exist in each file location
    def _contains_dicom_files(file_location):
        dicoms_path = os.path.join(input_dir, file_location)
        if pd.isna(file_location) or not os.path.isdir(dicoms_path):
            return False
        return len(os.listdir(dicoms_path)) > 0
    
    # Filter only rows where DICOM files exist in the corresponding location
    merged_df['Has DICOM Files'] = merged_df['File Location'].apply(_contains_dicom_files)
    merged_df = merged_df[merged_df['Has DICOM Files']]
    merged_df.drop(columns=['Has DICOM Files'], inplace=True)  # Remove the helper column

    return merged_df

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
def convert_dicom_to_cxr(patient_id, filtered_df, input_dir, output_dir):
    try:
        nlst_location = filtered_df.loc[filtered_df['pid'] == patient_id, 'File Location'].values[0]
        dicom_dir = os.path.join(os.path.dirname(input_dir), nlst_location)

        path_parts = Path(nlst_location).parts
        patient, experiment, instance = path_parts[1], path_parts[2], path_parts[-1]
        assert int(patient) == patient_id
        instance = instance.replace('.', '_')
        image_name = f'{experiment}__{instance}' # __ is the separator for the experiment and the instance name

        # Create output directory structure
        nifti_output_path = os.path.join(output_dir, 'preprocessed_xray_mha', patient)
        rgb_output_path = os.path.join(output_dir, 'preprocessed_xray_rgb', patient)
        os.makedirs(nifti_output_path, exist_ok=True)
        os.makedirs(rgb_output_path, exist_ok=True)

        # Convert DICOM files to NIfTI format and save
        nifti_output_path = os.path.join(nifti_output_path, f'{image_name}.mha')
        rgb_output_path = os.path.join(rgb_output_path, f'{image_name}.rgb')

        # start real processing here.

        dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
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
        slice_spacing = np.abs(dicom_slices[1].ImagePositionPatient[2] - dicom_slices[0].ImagePositionPatient[2])
        z_positions = [ds.ImagePositionPatient[2] for ds in dicom_slices]
        screening_length = abs(max(z_positions) - min(z_positions))  # Total Z-range

        # preprocess according to the Deep learning predicts cardiovascular disease risks from lung cancer screening low dose computed tomography paper.
        if slice_spacing > 3.0 or screening_length <= 200:
            return False

        # Calculate Z-spacing
        z_positions = [float(s.ImagePositionPatient[2]) for s in dicom_slices]
        z_spacing = np.abs(np.diff(z_positions).mean()) if len(z_positions) > 1 else slice_thickness

        nifti_image = nib.Nifti1Image(pixel_data, affine)
        img_data = nifti_image.get_fdata()

        # remove the defected vols (unreadable vols)
        if len(img_data.shape) != 3 or img_data.shape[0] == 1 or z_spacing == 0 or z_spacing == 0.0:
            return False

        #NOTE: rotate the axis so that it matches the ct orientation of the ct-rate dataset
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
        print(f"Error processing {dicom_dir}: {e}")
        return False


def main():

    # Directory paths
    input_dir = '/mnt/g/NLST/manifest-NLST_allCT/NLST'
    output_dir = '/mnt/g/NLST/manifest-NLST_allCT/temp' # testing purpose

    # Ensure output directory is created
    os.makedirs(output_dir, exist_ok=True)

    # Number of unique patients to sample
    filtered_df = filter_split(input_dir=os.path.dirname(input_dir), split='TEST')
    num_samples = filtered_df.shape[0]

    # Get list of patient IDs in the input directory
    patients = list(filtered_df['pid'])

    # Iterate over each sampled patient
    total_processed = 0

    progress_bar = tqdm(total=num_samples, desc="Processing Patients")

    for patient_id in patients:
        results = convert_dicom_to_cxr(patient_id, filtered_df, input_dir, output_dir)
        if results:
            # update the progress only when a patient is successfully processed
            total_processed += 1
            progress_bar.update(1)

        if total_processed >= num_samples:
            break

if __name__ == "__main__":
    # main() # NOTE: test bed without multiworkers
    
    # Directory paths
    input_dir = '/mnt/g/NLST/manifest-NLST_allCT/NLST'
    output_dir = '/mnt/g/NLST/manifest-NLST_allCT/preprocessed_NLST'

    # Ensure output directory is created
    os.makedirs(output_dir, exist_ok=True)

    # Number of unique patients to sample
    filtered_df = filter_split(input_dir=os.path.dirname(input_dir), split='ALL') #NOTE: ALL might takes longer to process
    num_samples = filtered_df.shape[0]

    # Get list of patient IDs in the input directory
    patients = list(filtered_df['pid'])

    num_workers = 8  # Number of worker processes

    # Process files using multiprocessing with tqdm progress bar
    with Pool(num_workers) as pool:
        func_with_arg = partial(convert_dicom_to_cxr, filtered_df=filtered_df, input_dir=input_dir, output_dir=output_dir)
        list(tqdm(pool.imap_unordered(func_with_arg, patients), total=len(patients)))
