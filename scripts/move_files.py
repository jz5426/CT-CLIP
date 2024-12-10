import os
import shutil
import torch

def move_files_preserve_structure(src_dir, dest_dir):
    """
    Moves files from src_dir to dest_dir while preserving the directory structure.
    Only moves files if they don't already exist in the destination.

    :param src_dir: Path to the source directory
    :param dest_dir: Path to the destination directory
    """
    total = 0
    for root, dirs, files in os.walk(src_dir):
        # Compute the relative path from the source root
        relative_path = os.path.relpath(root, src_dir)
        
        # Define the corresponding path in the destination
        dest_path = os.path.join(dest_dir, relative_path)
        
        # Create the destination directory if it doesn't exist
        os.makedirs(dest_path, exist_ok=True)
        
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            
            # Move the file only if it doesn't already exist in the destination
            if not os.path.exists(dest_file):
                shutil.copy2(src_file, dest_file)
                print(f"Copied: {src_file} -> {dest_file}")
            else:
                print(f"Skipped (already exists): {dest_file}")

def count_mha_files(directory):
    """
    Counts the total number of `.mha` files in the given directory, including subdirectories.

    :param directory: Path to the directory to search
    :return: Total number of `.mha` files
    """
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mha"):
                count += 1
    return count


# Example usage
# src_directory = "F:\\Chris\\dataset\\valid_preprocessed_xray_mha"  # Replace with the path to your source directory
# dest_directory = "F:\\Chris\\CT-RATE-FINAL\\processed_dataset\\valid_preprocessed_xray_mha"  # Replace with the path to your destination directory

src_directory = "F:\\Chris\\CT-RATE-FINAL\\processed_dataset\\"  # Replace with the path to your destination directory
dest_directory = "G:\\Chris\\CT-RATE-FINAL-BACKUP\\processed_dataset\\"  # Replace with the path to your source directory

move_files_preserve_structure(src_directory, dest_directory)
# files = count_mha_files(dest_directory)

# #NOTE: make sure the file and the embedding have one-to-one correspondence.
# saving_path = "F:\\Chris\\CT-RATE-FINAL\\processed_dataset\\features_embeddings\\valid\\image_features.pth"
# img_feature_path = os.path.join("F:\\Chris\\CT-RATE-FINAL\\processed_dataset\\features_embeddings\\valid\\image_features.pth")
# text_feature_path = os.path.join("F:\\Chris\\CT-RATE-FINAL\\processed_dataset\\features_embeddings\\valid\\text_features.pth")
# if os.path.exists(img_feature_path):
#     image_features = torch.load(img_feature_path)
# # if os.path.exists(text_feature_path):
# #     text_features = torch.load(text_feature_path)
# print('total files, ',files)
# print('number of keys ',len(image_features.keys()))