import os
import random
import shutil
import pydicom
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

# Directory paths
input_dir = '/mnt/g/NLST/manifest-NLST_allCT/NLST'
output_dir = '/mnt/g/NLST/manifest-NLST_allCT/20_percent_subset'

# Number of unique patients to sample
num_samples = 5250

# Ensure output directory is created
os.makedirs(output_dir, exist_ok=True)

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
def convert_dicom_to_nifti(dicom_dir, nift_output_path, rgb_output_path):
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
    # Calculate Z-spacing
    z_positions = [float(s.ImagePositionPatient[2]) for s in dicom_slices]
    z_spacing = np.abs(np.diff(z_positions).mean()) if len(z_positions) > 1 else slice_thickness

    nifti_image = nib.Nifti1Image(pixel_data, affine)
    img_data = nifti_image.get_fdata()

    # remove the defected vols
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
    sitk.WriteImage(xray_image, nift_output_path)

    return True

# Get list of patient IDs in the input directory
patients = [p for p in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, p))]

# Randomly sample unique patient IDs
sampled_patients = random.sample(patients, num_samples)

# Iterate over each sampled patient
total_processed = 0

progress_bar = tqdm(total=num_samples, desc="Processing Patients")

for patient_id in patients:
    patient_path = os.path.join(input_dir, patient_id)

    # Get list of experiments within patient directory
    experiments = [exp for exp in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, exp))]
    if not experiments:
        continue

    # Randomly select an experiment
    selected_experiment = random.choice(experiments)
    experiment_path = os.path.join(patient_path, selected_experiment)

    # Get list of instances within the experiment directory
    instances = [inst for inst in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, inst))]
    if not instances:
        continue

    # Select the instance with the most DICOM slices
    max_slices = 0
    selected_instance = None

    for inst in instances:
        instance_path = os.path.join(experiment_path, inst)
        dicom_files = [f for f in os.listdir(instance_path) if f.lower().endswith('.dcm')]

        if len(dicom_files) > max_slices:
            max_slices = len(dicom_files)
            selected_instance = inst

    if selected_instance is None:
        continue

    instance_path = os.path.join(experiment_path, selected_instance)

    # Create output directory structure
    nifti_output_path = os.path.join(output_dir, 'preprocessed_xray_mha', patient_id, selected_experiment, selected_instance)
    rgb_output_path = os.path.join(output_dir, 'preprocessed_xray_rgb', patient_id, selected_experiment, selected_instance)
    sampled_dicoms_output_path = os.path.join(output_dir, 'sampled_dicoms', patient_id, selected_experiment, selected_instance)
    os.makedirs(nifti_output_path, exist_ok=True)
    os.makedirs(rgb_output_path, exist_ok=True)

    # Convert DICOM files to NIfTI format and save
    nifti_output_path = os.path.join(nifti_output_path, f'{selected_instance}.mha')
    rgb_output_path = os.path.join(rgb_output_path, f'{selected_instance}.rgb')
    results = convert_dicom_to_nifti(instance_path, nifti_output_path, rgb_output_path)

    if results:
        # save the corresponding dicom images to the output directory
        os.makedirs(sampled_dicoms_output_path, exist_ok=True)
        # Copy all the DICOM slices from instance_path to sampled_dicoms_output_path
        for dicom_file in os.listdir(instance_path):
            if dicom_file.lower().endswith('.dcm'):
                shutil.copy(os.path.join(instance_path, dicom_file), os.path.join(sampled_dicoms_output_path, dicom_file))

        # update the progress only when a patient is successfully processed
        total_processed += 1
        progress_bar.update(1)

    if total_processed >= num_samples:
        break

print(f"Total {len(sampled_patients)} unique patient instances converted to NIfTI and copied to {output_dir}")
