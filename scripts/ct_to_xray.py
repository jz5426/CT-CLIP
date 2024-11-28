import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

def convert_ct_to_xray(path, title, projection_axis=1):
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

    mean_projection_filter = sitk.MeanProjectionImageFilter()
    mean_projection_filter.SetProjectionDimension(projection_axis)
    xray_image = mean_projection_filter.Execute(ct_image)
    xray_array = sitk.GetArrayFromImage(xray_image)

    xray_array = sitk.GetArrayFromImage(xray_image)
    xray_array = np.rot90(np.squeeze(xray_array), k=2)
    plt.imshow(xray_array, cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    plt.show()
    plt.savefig('./projected_xray/{}.png'.format(title))