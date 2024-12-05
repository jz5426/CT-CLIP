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
from zero_shot import CTClipInference

from data_preprocess.utils import convert_ct_to_xray

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


    # perform feature extraction after download 100 of them
    feature_extraction_frequency = 1000

    repo_id = "ibrahimhamamci/CT-RATE"

    # folder_path = "dataset/valid"
    folder_path = "dataset/train"

    # List all files in the repository
    all_files = list_repo_files(repo_id, repo_type="dataset")

    # Filter files in the 'valid' folder
    files = [f for f in all_files if f.startswith(folder_path)]

    # print(f"Files in the 'valid' folder: {valid_files}")
    print(f"Files in the 'train' folder: {files}")

    destination_folder = 'F:\\Chris\\CT-RATE'
    os.makedirs(destination_folder, exist_ok=True)


    tracking = 1
    for file in tqdm(files): # only download 1000 files (500MB x 1000) otherwise too large

        while tracking <= feature_extraction_frequency:
            # Extract the filename from the path
            filename = os.path.basename(file)
            dir_path = os.path.dirname(file)
            
            # Normalize each component to use consistent separators
            destination_folder = os.path.normpath(destination_folder)
            dir_path = os.path.normpath(dir_path)
            
            # Ensure the destination directory exists
            destination_dir = os.path.join(destination_folder, dir_path)
            os.makedirs(destination_dir, exist_ok=True)

            # Move the downloaded file to the destination folder
            file_dir = os.path.join(destination_dir, filename)

            #NOTE prevent repeated download.
            if os.path.exists(file_dir):
                continue

            # Download the file
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=dir_path.replace('\\', '/'),
                repo_type="dataset",
                token='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' # Ensure your token is set in the huggingface
            )
            shutil.move(file_path, file_dir)
            tracking += 1
        
        # TODO: preprocess the downloaded files and save the corresponding xray

        # TODO: remove the raw CT files and keep the xray files

        # start to do feature extraction TODO: change the paths
        inference = CTClipInference(
            clip,
            data_folder = "/mnt/f/Chris/dataset/train_preprocessed_ct",
            reports_file= "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/train_reports.csv",
            labels = "/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv",
            batch_size = 1,
            results_folder="inference_zeroshot/",
            num_train_steps = 1,
            feature_extraction_mode = True # extract only the text and ct features only
        )

        # feature extraction save the features to the phe object TODO: change the paths
        inference.feature_extraction('/mnt/f/Chris/dataset/features_embeddings', 'train')

    print("Finished")
