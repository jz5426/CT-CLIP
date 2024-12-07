from huggingface_hub import hf_hub_download, list_repo_files
import os
import shutil
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import torch
from pathlib import Path

# repo_id = "ibrahimhamamci/CT-RATE"
# # folder_path = "dataset/valid"
# folder_path = "dataset/train"

# # List all files in the repository
# all_files = list_repo_files(repo_id, repo_type="dataset")

# # Filter files in the 'valid' folder
# valid_files = [f for f in all_files if f.startswith(folder_path)]

# # print(f"Files in the 'valid' folder: {valid_files}")
# print(f"Files in the 'train' folder: {valid_files}")

# destination_folder = 'F:\\Chris\\CT-RATE'
# os.makedirs(destination_folder, exist_ok=True)

# for file in tqdm(valid_files[:3000]): # only download 1000 files (500MB x 1000) otherwise too large
#     # Extract the filename from the path
#     filename = os.path.basename(file)
#     dir_path = os.path.dirname(file)
    
#     # Normalize each component to use consistent separators
#     destination_folder = os.path.normpath(destination_folder)
#     dir_path = os.path.normpath(dir_path)
    
#     # Ensure the destination directory exists
#     destination_dir = os.path.join(destination_folder, dir_path)
#     os.makedirs(destination_dir, exist_ok=True)

#     # Move the downloaded file to the destination folder
#     file_dir = os.path.join(destination_dir, filename)

#     #NOTE check if folder exists before downloading next time
#     if os.path.exists(file_dir):
#         continue

#     # Download the file
#     file_path = hf_hub_download(
#         repo_id=repo_id,
#         filename=filename,
#         subfolder=dir_path.replace('\\', '/'),
#         repo_type="dataset",
#         token='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' # Ensure your token is set in the huggingface
#     )
#     shutil.move(file_path, file_dir)

# print(f"Downloaded files are saved in {destination_folder}")

def download_and_move(file, destination_folder, repo_id):
    # Extract the filename from the path
    filename = os.path.basename(file)
    dir_path = os.path.dirname(file)
    
    # Normalize each component to use consistent separators
    dir_path = os.path.normpath(dir_path)
    
    # Ensure the destination directory exists
    destination_dir = os.path.join(destination_folder, dir_path)
    os.makedirs(destination_dir, exist_ok=True)

    # Move the downloaded file to the destination folder
    file_dir = os.path.join(destination_dir, filename)


    # Download the file
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=dir_path.replace('\\', '/'), # for problems in windows machine
        repo_type="dataset",
        cache_dir='F:\\cache', #'/mnt/f/cache', # must be placed in the same external hard drive for wsl to work for shutil.move operations
        token='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' # Ensure your token is set in the huggingface
    )

    shutil.move(file_path, file_dir)

def parallel_download(batch, destination_folder, repo_id, num_workers=8):
    with Pool(num_workers) as pool:
        # Prepare the partial function with fixed arguments
        func_with_args = partial(download_and_move, destination_folder=destination_folder, repo_id=repo_id)
        # Process the files in parallel with a progress bar
        list(tqdm(pool.imap_unordered(func_with_args, batch), total=len(batch)))
        pool.close()

if __name__ == '__main__':

    split = 'train'

    # perform feature extraction after download 100 of them
    feature_extraction_frequency = 6000

    repo_id = "ibrahimhamamci/CT-RATE"
    folder_path = "dataset/{}".format(split)

    # List all files in the repository
    all_files = list_repo_files(repo_id, repo_type="dataset")

    # Filter files in the folder
    files = [f for f in all_files if f.startswith(folder_path)]

    # Filter out the files that are already processed previously
    saving_path = os.path.join('F:\\Chris\\dataset\\features_embeddings', split) # /mnt/f/Chris/dataset/features_embeddings'
    img_feature_path = os.path.join(saving_path, 'image_features.pth')
    text_feature_path = os.path.join(saving_path, 'text_features.pth')
    if os.path.exists(img_feature_path):
        image_features = torch.load(img_feature_path)
    if os.path.exists(text_feature_path):
        text_features = torch.load(text_feature_path)
    assert text_features.keys() == image_features.keys()
    keys = set(text_features.keys())
    files = [f for f in files if Path(Path(f).stem).stem not in keys] # nested path.stem due to .nii.gz, each remove one extension

    print(f"Files in the '{split}' folder: {len(files)}")
    total_files = len(files)

    destination_folder = os.path.normpath('F:\\Chris\\CT-RATE-temp') # F:\\Chris\\CT-RATE-temp\\dataset\\train /mnt/f/Chris/CT-RATE-temp
    os.makedirs(destination_folder, exist_ok=True)

    print('    downloading files\n')
    batch = files[:feature_extraction_frequency]
    parallel_download(batch, destination_folder, repo_id, num_workers=16)