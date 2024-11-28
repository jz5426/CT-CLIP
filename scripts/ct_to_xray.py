import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from torch.utils.data import DataLoader
import tqdm
from data_inference import CTReportDatasetinfer
from zero_shot import cycle
import os

def convert_ct_to_xray(path, title, projection_axis, target_path):
    """
    path: string path to the ct image
    title: title of the image being saved
    projection_axis: Axes: 0=depth, 1=height, 2=width
    """
    ct_image = sitk.ReadImage(path)

    # check spacing
    original_spacing  = ct_image.GetSpacing()

    # resample to the right voxel sapcing if not.
    if original_spacing != (1.0, 1.0, 1.0):
        original_size = ct_image.GetSize()
        direction = ct_image.GetDirection()
        origin = ct_image.GetOrigin()

        # Define the target spacing
        target_spacing = (1.0, 1.0, 1.0)

        # Calculate the new size to maintain the same physical dimensions
        new_size = [
            int(np.round(original_size[i] * (original_spacing[i] / target_spacing[i])))
            for i in range(3)
        ]

        # Resample the image
        ct_image = sitk.Resample(
            ct_image,
            new_size,
            sitk.Transform(),
            sitk.sitkLinear,  # Use sitk.sitkNearestNeighbor for label data
            origin,
            target_spacing,
            direction,
            0,  # Default pixel value for areas outside the original image
            ct_image.GetPixelID(),
        )   

    # project the ct image to xray
    mean_projection_filter = sitk.MeanProjectionImageFilter()
    mean_projection_filter.SetProjectionDimension(projection_axis)
    xray_image = mean_projection_filter.Execute(ct_image)
    xray_array = sitk.GetArrayFromImage(xray_image)
    xray_array = np.rot90(np.squeeze(xray_array), k=2)
    plt.imshow(xray_array, cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    saving_path = os.path.join(target_path, '{}.png'.format(title))

    directory_path = os.path.dirname(saving_path)
    # Create the directories if they don't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    plt.savefig(saving_path)

def ct_to_xrays(data_folder, reports_file, labels, target_path='./projected_xray'):
    ds = CTReportDatasetinfer(data_folder=data_folder, csv_file=reports_file, labels=labels, probing_mode=True)

    dl = DataLoader(ds, num_workers=1, batch_size=1,shuffle = True)
    # prepare with accelerator
    dl_iter=cycle(dl)

    for _ in tqdm.tqdm(range(len(ds))):
        _, _, _, acc_name, nii_file = next(dl_iter)
        
        # construct saving path
        acc_name = acc_name[0]
        nii_file = nii_file[0]
        paths = nii_file.split(os.sep)
        dirs = paths[-4:-1]
        filename = paths[-1].split('.')[0]
        dest = os.path.join(target_path, *dirs)

        #NOTE: only convert 1 axis for now, takes too long to load them one by one.
        # convert_ct_to_xray(nii_file, filename+'_axis0', 0, target_path)
        convert_ct_to_xray(nii_file, filename+'_axis1', 1, dest)
        # convert_ct_to_xray(nii_file, filename+'_axis2', 2, target_path)

if __name__ == '__main__':
    # data_folder='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/valid/'
    # reports_file="/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/dataset_radiology_text_reports_validation_reports.csv"
    # labels="/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
    # data_folder='C:/Users/MaxYo/OneDrive/Desktop/MBP/chris/CT-CLIP/dataset/valid'
    data_folder = 'C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\dataset\\valid'
    reports_file='C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\dataset\\radiology_text_reports\\dataset_radiology_text_reports_validation_reports.csv'
    labels='C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\dataset\\multi_abnormality_labels\\dataset_multi_abnormality_labels_valid_predicted_labels.csv'
    target_path='.\\projected_xray'
    ct_to_xrays(data_folder, reports_file, labels, target_path)