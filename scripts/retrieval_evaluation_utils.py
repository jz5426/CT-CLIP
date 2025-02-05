import os
from cxr_clip_utils import convert_dictconfig_to_dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIPwithXray
import random
import numpy as np
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from zero_shot import CTClipInference
import pandas as pd

def find_top_k_indices(values, k):
    # Check if the list has at least 50 values
    if len(values) < k:
        raise ValueError(f"The list must contain at least {k} values")

    # Use a combination of 'sorted' and 'enumerate' to sort the values while keeping track of indices
    sorted_values_with_indices = sorted(enumerate(values), key=lambda x: x[1], reverse=True)

    # Extract the indices of the top k values
    top_k_indices = [index for index, value in sorted_values_with_indices[:k]]

    return top_k_indices

def calc_similarity(arr1, arr2):
    oneandone = 0
    oneorzero = 0
    zeroandzero = 0
    for k in range(len(arr1)):
        if arr1[k] == 0 and arr2[k] == 0:
            zeroandzero += 1
        if arr1[k] == 1 and arr2[k] == 1:
            oneandone += 1
        if arr1[k] == 0 and arr2[k] == 1:
            oneorzero += 1
        if arr1[k] == 1 and arr2[k] == 0:
            oneorzero += 1

    return (oneandone / (oneandone + oneorzero))

def map_retrieval_evaluation(
      query_latents, # dictionary of the xray latents
      target_latents, # xray or CT feature dictionary
      metric_results_dest = "path_to_save_the_metric_results",
      predicted_label_csv_path='path_to_valid_predicted_labels.csv',
      k_list=[1, 5, 10, 50, 100],
      batch_size=1024,
      file_name='xray2ct',
      dataset='ct-rate' # ct-rate => internal, mimic => external, radchestct => external
    ):

    # convert the xray key as the accession to access the label later on.
    image_data_list = []
    accs = []
    for xray_file_key in tqdm.tqdm(query_latents.keys()):
        image_data_list.append(query_latents[xray_file_key]) # insert the embeddings

        if dataset == 'ct-rate':
            accs.append(xray_file_key+'.nii.gz')  # Use the filename without the extension as the accession number
        elif dataset == 'mimic':
            accs.append(xray_file_key)  # Use the filename without the extension as the accession number

    # Concatenate all loaded image data
    image_data = np.array(image_data_list)
    print(image_data.shape)

    # mainly for reading the file labels.
    df = pd.read_csv(predicted_label_csv_path)

    running_ratios_external = []
    image_data_for_second = []
    accs_for_second = []
    # Filter the image data based on the condition in the validation labels
    for target_key in tqdm.tqdm(target_latents.keys()):
        if dataset == 'ct-rate':
            acc_second = target_key+'.nii.gz'
            row_second = df[df['VolumeName'] == acc_second]
        elif dataset == 'mimic':
            acc_second = target_key
            row_second = df[df['hadm_id'] == acc_second]

        num_path = np.sum(row_second.iloc[:, 1:].values[0])

        # if there are any labels (multihot or onehot) for this, save the embeddings and the file name NOTE: do we need this?
        if num_path != 0:
            target_latent = target_latents[target_key]
            image_data_for_second.append(target_latent)
            accs_for_second.append(acc_second)
    
    # one huge matrix
    image_data_for_second = np.array(image_data_for_second)
    print(image_data_for_second.shape)

    list_outs = []
    # Calculate the similarity for each image in the dataset
    for return_n in k_list:
        ratios_external = [] # take note for this one.
        for i in tqdm.tqdm(range(image_data.shape[0])):
            first = image_data[i] # get the embedding
            first = torch.tensor(first).to('cuda') # place it in the GPU for batch processing.
            acc_first = accs[i]
            if dataset == 'ct-rate':
                row_first = df[df['VolumeName'] == acc_first]
            elif dataset == 'mimic':
                row_first = df[df['hadm_id'] == acc_first]
            row_first = row_first.iloc[:, 1:].values[0]

            # Create a DataLoader for batching processing, with respect to each row_first
            dataset = TensorDataset(torch.tensor(image_data_for_second))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            crosses = []
            ratios_internal = []
            for batch in dataloader:
                second = batch[0].to('cuda')
                cross_batch = torch.matmul(first, second.T)
                crosses.extend(cross_batch.cpu().tolist())

            top_k_indices = find_top_k_indices(crosses, return_n)
            for index in top_k_indices:
                acc_second = accs_for_second[index]
                if dataset == 'ct-rate':
                    row_second = df[df['VolumeName'] == acc_second]
                elif dataset == 'mimic':
                    row_second = df[df['hadm_id'] == acc_second]
                row_second = row_second.iloc[:, 1:].values[0]

                # find the similarity (overlapping labels) based on the top-k
                ratio = calc_similarity(row_first, row_second)
                ratios_internal.append(ratio)
            running_ratios_external.append(np.mean(np.array(ratios_internal)))
            ratios_external.append(np.mean(np.array(ratios_internal)))

        running_avg_stats = str(np.mean(np.array(running_ratios_external)))
        stats = str(np.mean(np.array(ratios_external)))

        print(running_avg_stats, stats)
        list_outs.append(str((running_avg_stats, stats)))

    # output_file_path = data_folder + f"internal_accessions_t2i_{list_ks[0]}.txt"
    output_file_path = metric_results_dest + f"{file_name}.txt"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Open the file for writing (you can also use "a" to append if the file already exists)
    with open(output_file_path, "w") as file:
        # Write each string from the list to the file
        for string in list_outs:
            file.write(string + "\n")
    print(f'results saved to {output_file_path}')

    # TODO: eventually save the metrics to the excel spreadsheets
    return list_outs

def recall_retrieval_evaluation(
        query_latents, 
        target_latents, 
        list_ks=[5, 10, 50, 100], 
        data_folder = "",
        file_name='xray2ct',
        batch_size=1024,
        dataset = 'ct-rate'):

    query_latents = np.array(query_latents)
    target_latents = np.array(target_latents) # to be retrieved from

    list_texts = []
    for value in tqdm.tqdm(list_ks):
        num_is_in, num_random = 0, 0

        # for each xray => the goal is to retrieve the correct target
        for i in tqdm.tqdm(range(query_latents.shape[0])):
            crosses, crosses_rands = [], []
            xray = torch.tensor(query_latents[i]).to('cuda')

            # Create a DataLoader for batching
            dataset = TensorDataset(torch.tensor(target_latents))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # find the similarity between the xray and the target embeddings
            for batch in dataloader:
                targets = batch[0].to('cuda')
                
                # Compute similarity in batch and save the results.
                cross_batch = torch.matmul(xray, targets.T)
                crosses.extend(cross_batch.cpu().tolist())

            # find the similarity between the xray and the target embeddings
            for batch in dataloader:
                targets = batch[0].to('cuda')
                
                # Compute similarity in batch and save the results.
                cross_batch = torch.matmul(xray, targets.T)
                crosses.extend(cross_batch.cpu().tolist())
            
            # find the top k indiices
            top_k_indices = find_top_k_indices(crosses, value)
            if i in top_k_indices:
                num_is_in += 1

        clip = num_is_in / target_latents.shape[0]
        # rand = num_random / target_latents.shape[0]
        rand = 'n.a'
        write_str = f"K={value}, clip = {clip}, rand= {rand}"
        list_texts.append(write_str)

    # output_file_path = data_folder + f"internal_accessions_t2i_{list_ks[0]}.txt"
    output_file_path = data_folder + f"{file_name}.txt"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Open the file for writing (you can also use "a" to append if the file already exists)
    with open(output_file_path, "w") as file:
        # Write each string from the list to the file
        for string in list_texts:
            file.write(string + "\n")
    print(f'results saved to {output_file_path}')

    # TODO: eventually save the metrics to the excel spreadsheets
    return list_texts


def ctrate_retrieval_evaluation():
    """list all the retrieval evaluation for ct-rate dataset"""

    split = 'valid'
    embedding_directory = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/'
    # get the preprocessed image and text features
    # NOTE: these are normalized features
    saving_path = os.path.join(embedding_directory, split)
    img_feature_path = os.path.join(saving_path, 'image_features.pth')
    text_feature_path = os.path.join(saving_path, 'text_features.pth')
    image_features, text_features = None, None
    if os.path.exists(img_feature_path):
        image_features = torch.load(img_feature_path)
    if os.path.exists(text_feature_path):
        text_features = torch.load(text_feature_path)
    assert(image_features.keys() == text_features.keys())
    ct_report_embeddings = [(image_features[key], text_features[key]) for key in image_features.keys()]

    ## the following are the upper baseline from CT-CLIP

    # report2ct
    print('evaluating report 2 ct in recall')
    recall_retrieval_evaluation(
        query_latents=[embed[1] for embed in ct_report_embeddings],
        target_latents=[embed[0].reshape(-1) for embed in ct_report_embeddings],
        file_name='report2ct_recall')

    # ct2report
    print('evaluating ct 2 report in recall')
    recall_retrieval_evaluation(
        query_latents=[embed[0] for embed in ct_report_embeddings],
        target_latents=[embed[1].reshape(-1) for embed in ct_report_embeddings],
        file_name='ct2report_recall',
        dataset='ct-rate'
    )

    print('evaluating report 2 ct in MAP')
    map_retrieval_evaluation(
        text_features,
        target_latents=image_features,
        predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        file_name='report2ct_map',
        dataset='ct-rate'
    )

    # ct2ct
    print('evaluating ct 2 ct in MAP')
    map_retrieval_evaluation(
        image_features,
        target_latents=image_features,
        predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        file_name='ct2ct_map',
        dataset='ct-rate'
    )

    print('evaluating ct 2 report in MAP')
    map_retrieval_evaluation(
        image_features,
        target_latents=text_features,
        predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        file_name='ct2report_map',
        dataset='ct-rate'
    )
    
    print('evaluating report 2 report in MAP')
    map_retrieval_evaluation(
        text_features,
        target_latents=text_features,
        predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        file_name='report2report_map',
        dataset='ct-rate'
    )

def mimic_retrieval_evaluation():
    """list all the retrieval evaluation for the mimic dataset"""
    return