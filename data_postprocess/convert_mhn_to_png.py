import os
import SimpleITK as sitk
import numpy as np
from PIL import Image

def convert_2d_mha_to_png(source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(".mha"):
            # Read 2D .mha image
            mha_path = os.path.join(source_dir, filename)
            image = sitk.ReadImage(mha_path)
            array = sitk.GetArrayFromImage(image)

            # # Normalize to 0–255
            # array = (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-8)
            # array = (array * 255).astype(np.uint8)

            np_image = (array - array.min()) / (array.max() - array.min()) * 255
            np_image = np_image.astype(np.uint8)  # Convert to uint8 for PIL compatibility
            rgb_image = np.stack([np_image] * 3, axis=-1)  # Shape: (H, W, 3)
            # rgb_image = Image.fromarray(rgb_image, mode="RGB")

            # Save as PNG
            img = Image.fromarray(rgb_image)
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(dest_dir, png_filename)
            img.save(png_path)

    print(f"Converted all .mha files from {source_dir} to PNGs in {dest_dir}")

# Example usage
source_directory = "/cluster/projects/mcintoshgroup/publicData/CT-RATE/coronary_artery_wall_cal_visualize/CT-RATE"
destination_directory = "/cluster/projects/mcintoshgroup/publicData/CT-RATE/coronary_artery_wall_cal_visualize/CT-RATE_png"
convert_2d_mha_to_png(source_directory, destination_directory)
