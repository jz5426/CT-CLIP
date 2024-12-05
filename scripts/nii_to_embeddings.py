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
# from zero_shot import CTClipInference
import accelerate
import preprocess_utils
from zero_shot import CTClipInference
from pathlib import Path

if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    text_encoder.resize_token_embeddings(len(tokenizer))

    image_encoder = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 480,
        patch_size = 20,
        temporal_patch_size = 10,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    )

    clip = CTCLIP(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_image = 294912,
        dim_text = 768,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False
    )

    clip.load("/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/models/CT-CLIP_v2.pt")

    split = 'train'

    # perform feature extraction after download 100 of them
    feature_extraction_frequency = 2

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

    print(f"Files in the '{split}' folder: {len(files)}")
    total_files = 5 #len(files)

    destination_folder = os.path.normpath('/mnt/f/Chris/CT-RATE-temp')
    os.makedirs(destination_folder, exist_ok=True)

    for i in tqdm(range(0, total_files, feature_extraction_frequency)):
        batch = files[i:i + feature_extraction_frequency]

        for file in batch:
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

        print('    processing raw ct files')
        # NOTE: preprocess the downloaded files and save the corresponding xray
        raw_ct_path = os.path.join(destination_folder, 'dataset', f'{split}') # load the data for processing from here
        processed_ct_dest = os.path.join(destination_folder, 'processed_dataset') # destination folder to hold the processed ct and xray files
        preprocess_utils.process(nii_path=raw_ct_path, shared_dest=processed_ct_dest, split=split)

        # NOTE: remove the raw CT files and keep the xray files
        print('    removing raw ct files')
        shutil.rmtree(raw_ct_path)

        # start to do feature extraction
        processed_ct_directory = os.path.join(processed_ct_dest, f"{split}_preprocessed_ct")
        inference = CTClipInference(
            clip,
            data_folder = processed_ct_directory,
            reports_file = f"/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/{split}_reports.csv",
            labels = f"/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv",
            batch_size = 1,
            results_folder="inference_zeroshot/",
            num_train_steps = 1,
            feature_extraction_mode = True # extract only the text and ct features only
        )

        # feature extraction save the features to the phe object
        print('    performing feature extraction')
        inference.feature_extraction('/mnt/f/Chris/dataset/features_embeddings', f'{split}')

        #NOTE: remove the preprocessed ct files ONLY
        print('    removing processed ct files')
        shutil.rmtree(processed_ct_directory)
        
    print("Finished")