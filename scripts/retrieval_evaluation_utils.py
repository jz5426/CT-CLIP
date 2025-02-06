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
from eval_utils import metadata_base_on_model_type
from zero_shot import CTClipInference, MimicCTClipInference
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
      metric_results_dest = "",
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
        elif dataset == 'radchest_ct':
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
        elif dataset == 'radchest_ct':
            acc_second = target_key
            row_second = df[df['NoteAcc_DEID'] == acc_second]

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
            elif dataset == 'radchest_ct':
                row_first = df[df['NoteAcc_DEID'] == acc_first]
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
                elif dataset == 'radchest_ct':
                    row_second = df[df['NoteAcc_DEID'] == acc_second]
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
        metric_results_dest = "",
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
            train_dataset = TensorDataset(torch.tensor(target_latents))
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

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
    output_file_path = metric_results_dest + f"{file_name}.txt"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Open the file for writing (you can also use "a" to append if the file already exists)
    with open(output_file_path, "w") as file:
        # Write each string from the list to the file
        for string in list_texts:
            file.write(string + "\n")
    print(f'results saved to {output_file_path}')

    # TODO: eventually save the metrics to the excel spreadsheets
    return list_texts


def ctrate_retrieval_evaluation(params):
    """list all the retrieval evaluation for ct-rate dataset"""

    cfg = params['cfg']
    baselines = params['baselines']
    image_encoder = params['image_encoder']
    text_encoder = params['text_encoder']
    tokenizer = params['tokenizer']
    metric_results_destination = params['metric_results_destination']

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
        metric_results_dest=metric_results_destination,
        file_name='report2ct_recall',
        train_dataset='ct-rate'
    )

    # ct2report
    print('evaluating ct 2 report in recall')
    recall_retrieval_evaluation(
        query_latents=[embed[0] for embed in ct_report_embeddings],
        target_latents=[embed[1].reshape(-1) for embed in ct_report_embeddings],
        metric_results_dest=metric_results_destination,
        file_name='ct2report_recall',
        train_dataset='ct-rate'
    )

    print('evaluating report 2 ct in MAP')
    map_retrieval_evaluation(
        text_features,
        target_latents=image_features,
        metric_results_dest=metric_results_destination,
        predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        file_name='report2ct_map',
        dataset='ct-rate'
    )

    # ct2ct
    print('evaluating ct 2 ct in MAP')
    map_retrieval_evaluation(
        image_features,
        target_latents=image_features,
        metric_results_dest=metric_results_destination,
        predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        file_name='ct2ct_map',
        dataset='ct-rate'
    )

    print('evaluating ct 2 report in MAP')
    map_retrieval_evaluation(
        image_features,
        target_latents=text_features,
        metric_results_dest=metric_results_destination,
        predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        file_name='ct2report_map',
        dataset='ct-rate'
    )
    
    print('evaluating report 2 report in MAP')
    map_retrieval_evaluation(
        text_features,
        target_latents=text_features,
        metric_results_dest=metric_results_destination,
        predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        file_name='report2report_map',
        dataset='ct-rate'
    )


    for baseline in baselines:

        dim_xray, xray_model_type, pth_name, latent_size = metadata_base_on_model_type(baseline)

        # automatically load the model weights
        clip_xray = CTCLIPwithXray(
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            dim_text = 768,
            dim_image = 294912,
            xray_model_type = xray_model_type,
            dim_xray = dim_xray,
            dim_latent = 512,
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False,
            cfg=cfg
        )

        # check the trainable parameters
        # xray_encoder_trainable = sum(p.numel() for p in clip_xray.xray_encoder.parameters() if p.requires_grad)
        # ct_clip_trainable = sum(p.numel() for p in clip_xray.CTCLIP.parameters() if p.requires_grad)
        # assert(xray_encoder_trainable == 0)
        # assert(ct_clip_trainable == 0)

        retrival_evaluator = CTClipInference(
            clip_xray,
            cfg=cfg,
            tokenizer=tokenizer,
            data_folder= f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/{split}_preprocessed_xray_mha',
            # NOTE: the embedding paths are MANDATORY for the dataloader to work. RUN THIS SCRIPT MAINLY AFTER THE CTCLIP EMBEDDINGS ARE EXTRACTED.
            img_embedding_paths = {
                f'{split}': f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/{split}/image_features.pth'
            },
            text_embedding_paths = {
                f'{split}': f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/{split}/text_features.pth'
            },
            reports_file = f'/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/{split}_reports.csv',
            labels = f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            results_folder="./inference_zeroshot_retrieval",
            batch_size = 512,
            num_train_steps = -1, # placeholder
            num_workers = 10, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
            feature_extraction_mode = True # might be optional
        )  

        # get xray latent features from a model NOTE: to be safe, re-extract the xray feature everytime
        xray_features = retrival_evaluator.xray_feature_extraction(embedding_directory, pth_name=pth_name, append=False)

        # make sure all three dictionary contains the same set of keys
        assert(image_features.keys() == text_features.keys() == xray_features.keys())

        # organize data into a list with index as a the text-image-xray correspondance and pair up xray-ct_image and xray-text
        triplet_embeddings = [(image_features[key], text_features[key], xray_features[key]) for key in xray_features.keys()]

        # NOTE: all features are normalized.

        print('evaluating xray 2 ct_volumes recall')
        recall_retrieval_evaluation(
            query_latents=[triple[-1] for triple in triplet_embeddings],
            target_latents=[triple[0].reshape(-1) for triple in triplet_embeddings],
            metric_results_dest=metric_results_destination,
            file_name=f'{baseline}_synxray2ct_recall',
            train_dataset='ct-rate')
        print('evaluating ct_volumes 2 xray recall')
        recall_retrieval_evaluation(
            query_latents=[triple[0] for triple in triplet_embeddings],
            target_latents=[triple[-1].reshape(-1) for triple in triplet_embeddings],
            metric_results_dest=metric_results_destination,
            file_name=f'{baseline}_ct2synxray_recall',
            train_dataset='ct-rate')

        print('evaluating xray 2 ct_reports recall')
        recall_retrieval_evaluation(
            query_latents=[triple[-1] for triple in triplet_embeddings],
            target_latents=[triple[1].reshape(-1) for triple in triplet_embeddings],
            metric_results_dest=metric_results_destination,
            file_name=f'{baseline}_synxray2report_recall',
            train_dataset='ct-rate')
        print('evaluating ct_reports 2 xray recall')
        recall_retrieval_evaluation(
            query_latents=[triple[1] for triple in triplet_embeddings],
            target_latents=[triple[-1].reshape(-1) for triple in triplet_embeddings],
            metric_results_dest=metric_results_destination,
            file_name=f'{baseline}_report2synxray_recall',
            train_dataset='ct-rate')



        print('evaluating xray 2 ct_volumes MAP')
        map_retrieval_evaluation(
            xray_features,
            target_latents=image_features,
            metric_results_dest=metric_results_destination,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{baseline}_synxray2ct_map',
            dataset='ct-rate')
        print('evaluating ct_volumes 2 xray MAP')
        map_retrieval_evaluation(
            image_features,
            target_latents=xray_features,
            metric_results_dest=metric_results_destination,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{baseline}_ct2synxray_map',
            dataset='ct-rate')



        print('evaluating xray 2 ct_reports MAP')
        map_retrieval_evaluation(
            xray_features,
            target_latents=text_features,
            metric_results_dest=metric_results_destination,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{baseline}_synxray2report_map',
            dataset='ct-rate')
        print('evaluating ct_reports 2 xray MAP')
        map_retrieval_evaluation(
            text_features,
            target_latents=xray_features,
            metric_results_dest=metric_results_destination,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{baseline}_report2synxray_map',
            dataset='ct-rate')



        # there is not symmetric retrieval and recall for this one.
        print('evaluating xray 2 xray MAP')
        map_retrieval_evaluation(
            xray_features,
            target_latents=xray_features,
            metric_results_dest=metric_results_destination,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{baseline}_synxray2synxray_map',
            dataset='ct-rate')

def radchest_ct_retrieval_evaluation(params):
    cfg = params['cfg']
    baselines = params['baselines']
    image_encoder = params['image_encoder']
    text_encoder = params['text_encoder']
    tokenizer = params['tokenizer']
    metric_results_destination = params['metric_results_destination']

    embedding_directory = '/cluster/projects/mcintoshgroup/publicData/RADChestCT/features_embeddings'
    saving_path = embedding_directory
    img_feature_path = os.path.join(saving_path, 'image_features.pth')
    image_features = None
    if os.path.exists(img_feature_path):
        image_features = torch.load(img_feature_path)

    for baseline in baselines:
        dim_xray, xray_model_type, pth_name, latent_size = metadata_base_on_model_type(baseline)

        clip_xray = CTCLIPwithXray(
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            dim_text = 768, # for ct-clip
            dim_image = 294912, # for ct-clip
            xray_model_type = xray_model_type,
            dim_xray = dim_xray,
            dim_latent = 512, # the target output latent dimension
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False,
            cfg=cfg,
            auto_load_pretrained_weights = True # NOTE: automatically load the model weights based on the xray_model_type
        )

        retrival_evaluator = CTClipInference(
            clip_xray,
            tokenizer=tokenizer,
            data_folder = '/cluster/projects/mcintoshgroup/publicData/RADChestCT/preprocessed_xray_mha',
            labels = '/cluster/projects/mcintoshgroup/publicData/RADChestCT/final_labels.csv',
            batch_size = 4,
            num_workers = 5,
            results_folder="inference_zeroshot/",
            num_train_steps = 1,
            feature_extraction_mode = True, # extract only the text and ct features only
            dataset='radchest_ct' # this is what differentiate with ct-rate one.
        )

        # get xray latent features from a model
        xray_features = retrival_evaluator.extract_xray_features()
        assert(image_features.keys() == xray_features.keys())

        triplet_embeddings = [(image_features[key], 'placeholder', xray_features[key]) for key in xray_features.keys()]

        print('evaluating xray 2 ct_volumes recall')
        recall_retrieval_evaluation(
            query_latents=[triple[-1] for triple in triplet_embeddings],
            target_latents=[triple[0].reshape(-1) for triple in triplet_embeddings],
            metric_results_dest=metric_results_destination,
            file_name=f'{baseline}_synxray2ct_recall',
            train_dataset='radchest_ct')
        print('evaluating ct_volumes 2 xray recall')
        recall_retrieval_evaluation(
            query_latents=[triple[0] for triple in triplet_embeddings],
            target_latents=[triple[-1].reshape(-1) for triple in triplet_embeddings],
            metric_results_dest=metric_results_destination,
            file_name=f'{baseline}_ct2synxray_recall',
            train_dataset='radchest_ct')


        print('evaluating xray 2 ct_volumes MAP')
        map_retrieval_evaluation(
            xray_features,
            target_latents=image_features,
            metric_results_dest=metric_results_destination,
            predicted_label_csv_path='/cluster/projects/mcintoshgroup/publicData/RADChestCT/final_labels.csv',
            file_name=f'{baseline}_synxray2ct_map',
            dataset='radchest_ct')
        print('evaluating ct_volumes 2 xray MAP')
        map_retrieval_evaluation(
            image_features,
            target_latents=xray_features,
            metric_results_dest=metric_results_destination,
            predicted_label_csv_path='/cluster/projects/mcintoshgroup/publicData/RADChestCT/final_labels.csv',
            file_name=f'{baseline}_ct2synxray_map',
            dataset='radchest_ct')



def mimic_retrieval_evaluation(params):
    """list all the retrieval evaluation for the mimic dataset"""
    cfg = params['cfg']
    baselines = params['baselines']
    image_encoder = params['image_encoder']
    text_encoder = params['text_encoder']
    tokenizer = params['tokenizer']
    metric_results_destination = params['metric_results_destination']

    for baseline in baselines:
        dim_xray, xray_model_type, pth_name, latent_size = metadata_base_on_model_type(baseline)

        clip_xray = CTCLIPwithXray(
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            dim_text = 768, # for ct-clip
            dim_image = 294912, # for ct-clip
            xray_model_type = xray_model_type,
            dim_xray = dim_xray,
            dim_latent = 512, # the target output latent dimension
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False,
            cfg=cfg,
            auto_load_pretrained_weights = True # NOTE: automatically load the model weights based on the xray_model_type
        )

        # check the trainable parameters
        # xray_encoder_trainable = sum(p.numel() for p in clip_xray.xray_encoder.parameters() if p.requires_grad)
        # ct_clip_trainable = sum(p.numel() for p in clip_xray.CTCLIP.parameters() if p.requires_grad)
        # assert(xray_encoder_trainable == 0)
        # assert(ct_clip_trainable == 0)
        
        retrival_evaluator = MimicCTClipInference(
            clip_xray,
            cfg=cfg,
            tokenizer=tokenizer,
            data_folder= '/cluster/home/t135419uhn/CT-CLIP/preprocessed_mimic/mimic_preprocessed_xray_mha',
            reports_file = '/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/external_valid_mimic_report.csv',
            labels = '/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_external_valid_mimic_labels.csv',
            results_folder="./inference_zeroshot_retrieval_mimic",
            batch_size = 512,
            num_workers = 2, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
            feature_extraction_mode = True # might be optional
        )  

        # get xray latent features from a model
        xray_features = retrival_evaluator.extract_xray_features()

        # get text features from the model.
        text_features = retrival_evaluator.extract_report_features()

        # NOTE: all features are normalized.

        # # organize data into a list with index as a the text-image-xray correspondance and pair up xray-ct_image and xray-text
        triplet_embeddings = [('', text_features[key], xray_features[key]) for key in xray_features.keys()]

        # the experiment is in the same order as the table listed in external_validation document in notion.

        print('evaluating xray 2 ct_report MAP')
        map_retrieval_evaluation(
            xray_features,
            target_latents=text_features,
            metric_results_dest=metric_results_destination,
            predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_external_valid_mimic_labels.csv',
            file_name=f'{baseline}_mimic_xray2report_map',
            dataset='mimic')

        print('evaluating xray 2 ct reports recall')
        recall_retrieval_evaluation(
            query_latents=[triple[-1] for triple in triplet_embeddings],
            target_latents=[triple[1].reshape(-1) for triple in triplet_embeddings],
            metric_results_dest=metric_results_destination,
            file_name=f'{baseline}_mimic_xray2report_recall',
            train_dataset='mimic')

        print('evaluating xray 2 xray MAP')
        map_retrieval_evaluation(
            xray_features,
            target_latents=xray_features,
            metric_results_dest=metric_results_destination,
            predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_external_valid_mimic_labels.csv',
            file_name=f'{baseline}_mimic_xray2mimic_xray_map',
            dataset='mimic')

        print('evaluating report 2 xray recall')
        recall_retrieval_evaluation(
            query_latents=[triple[1] for triple in triplet_embeddings],
            target_latents=[triple[-1].reshape(-1) for triple in triplet_embeddings],
            metric_results_dest=metric_results_destination,
            file_name=f'{baseline}_report2mimic_xray_recall',
            train_dataset='mimic')

        print('evaluating report 2 xray MAP')
        map_retrieval_evaluation(
            text_features,
            target_latents=xray_features,
            metric_results_dest=metric_results_destination,
            predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_external_valid_mimic_labels.csv',
            file_name=f'{baseline}_report2mimic_xray_map',
            dataset='mimic')

    return