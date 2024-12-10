"""
this script download the ct file one by one and transform the image to a embeddings and save it.
"""

from huggingface_hub import hf_hub_download, list_repo_files
import os
import shutil
from tqdm import tqdm

import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
import preprocess_utils
from zero_shot import CTClipInference
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import multiprocessing
from pathlib import Path

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
        subfolder=dir_path,
        repo_type="dataset",
        cache_dir='/mnt/f/cache', # must be placed in the same external hard drive for wsl to work for shutil.move operations
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
    # multiprocessing.set_start_method('spawn', force=True)
    split = 'train'

    # perform feature extraction after download 100 of them
    feature_extraction_frequency = 12000

    repo_id = "ibrahimhamamci/CT-RATE"
    folder_path = "dataset/{}".format(split)

    # List all files in the repository
    all_files = list_repo_files(repo_id, repo_type="dataset")

    # Filter files in the folder
    files = [f for f in all_files if f.startswith(folder_path)]

    # Filter out the files that are already processed previously
    saving_path = os.path.join('/mnt/f/Chris/dataset/features_embeddings', split)
    img_feature_path = os.path.join(saving_path, 'image_features.pth')
    text_feature_path = os.path.join(saving_path, 'text_features.pth')
    if os.path.exists(img_feature_path):
        image_features = torch.load(img_feature_path)
    if os.path.exists(text_feature_path):
        text_features = torch.load(text_feature_path)
    assert text_features.keys() == image_features.keys()
    keys = set(text_features.keys())
    files = [f for f in files if Path(Path(f).stem).stem not in keys] # nested path.stem due to .nii.gz, each remove one extension

    # list preprocessed xray files
    tmp_path_files = list(Path(f'/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/{split}_preprocessed_xray_mha').rglob('*.*'))  # Find all files (excluding directories)
    base_path_files = list(Path(f'/mnt/f/Chris/dataset/{split}_preprocessed_xray_mha').rglob('*.*'))  # Find all files (excluding directories)

    # files = list(Path(f'/mnt/f/Chris/dataset/{split}_preprocessed_xray_mha').rglob('*.*'))  # Find all files (excluding directories)
    filenames = set([os.path.basename(file)[:-len('.mha')] for file in tmp_path_files] + [os.path.basename(file)[:-len('.mha')] for file in base_path_files])
    print(f"files in the embedding dictionary {len(keys)}. files that are preprocessed {len(filenames)}")

    tmp_path_files_rgb = list(Path(f'/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/{split}_preprocessed_xray_rgb').rglob('*.*'))  # Find all files (excluding directories)
    base_path_files_rgb = list(Path(f'/mnt/f/Chris/dataset/{split}_preprocessed_xray_rgb').rglob('*.*'))
    filenames_rgb = set([os.path.basename(file)[:-len('.png')] for file in tmp_path_files_rgb] + [os.path.basename(file)[:-len('.png')] for file in base_path_files_rgb])

    # delete it
    for key in keys:
        if key not in filenames:
            del text_features[key]
            del image_features[key]
        
    print(f'number of embeddings {len(text_features)}, files that are preprocessed in .mha {len(filenames)}, files that are preprocessed in .png {len(filenames_rgb)}')
    # torch.save(image_features, img_feature_path)
    # torch.save(text_features, text_feature_path)
    
    print("Finished")
    # process(nii_path='/mnt/f/Chris/CT-RATE-FINAL/dataset/train', shared_dest='/mnt/f/Chris/CT-RATE-FINAL/processed_dataset', split='train', num_workers=1)
