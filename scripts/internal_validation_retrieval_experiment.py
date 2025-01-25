"""
largely extends from the synxray_to_report_ct.py
"""

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

def load_model_weights(clip_xray, cfg, ckpt_name=None):

    if ckpt_name == None: 
        # NOTE: random weights
        ckpt_name = 'random'
        pth_name = 'random_xray_features.pth'
        rand_layers = 0
        for layer in clip_xray.xray_encoder.modules():
            if hasattr(layer, 'reset_parameters'):
                rand_layers += 1
                layer.weight.data = torch.randn_like(layer.weight)
                if layer.bias is not None:
                    layer.bias.data = torch.randn_like(layer.bias)
        for layer in clip_xray.to_xray_latent.modules():
            if hasattr(layer, 'reset_parameters'):
                rand_layers += 1
                layer.weight.data = torch.randn_like(layer.weight)
                if layer.bias is not None:
                    layer.bias.data = torch.randn_like(layer.bias)
        print('Loaded ramdom weights')
        print(f'number of randomly initialized layers {rand_layers}')
    elif ckpt_name == 'cxr_clip':
        # NOTE: cxr-clip pretrained weights
        ckpt_name = 'r50_mcc' if cfg['model']['image_encoder']['name'] == 'resnet' else 'swint_mcc'
        clip_xray.load_cxr_clip_xray_encoder(
            '/cluster/home/t135419uhn/CT-CLIP/models/cxr_clip/{}.tar'.format(ckpt_name), # cxr-clip pretrained
            freeze_weights=True
        )
        pth_name = 'cxr_xray_features.pth'
        print('Loaded weights from cxr_clip')
    #TODO: add more baseline here
    else:
        # NOTE: our weights
        # ckpt_name='modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch'
        clip_xray.load_our_pretrained_weights(f'/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/{ckpt_name}.pt')
        pth_name = f'{ckpt_name}_xray_features.pth'
        print(f'Loaded weights from {ckpt_name}')

    return clip_xray, pth_name

def map_retrieval_evaluation(
        query_latents, # dictionary of the xray latents
        target_latents, # xray or CT feature dictionary
        data_folder = "./internal_val_retrieval_results/",
        predicted_label_csv_path='path_to_valid_predicted_labels.csv',
        k_list=[1,5,10,50, 100],
        batch_size=1024,
        file_name='xray2ct',
    ):

    # convert the xray key as the accession to access the label later on.
    image_data_list = []
    accs = []
    for xray_file_key in tqdm.tqdm(query_latents.keys()):
        image_data_list.append(query_latents[xray_file_key]) # insert the embeddings
        accs.append(xray_file_key+'.nii.gz')  # Use the filename without the extension as the accession number

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

        acc_second = target_key+'.nii.gz'
        row_second = df[df['VolumeName'] == acc_second]
        num_path = np.sum(row_second.iloc[:, 1:].values[0])

        # if there are any labels (multihot or onehot) for this, save the embeddings and the file name NOTE: do we need this?
        if num_path != 0:
            target_latent = target_latents[target_key]
            image_data_for_second.append(target_latent)
            accs_for_second.append(acc_second)
        # else:
        #     print(acc_second)
    
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
            row_first = df[df['VolumeName'] == acc_first]
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
                row_second = df[df['VolumeName'] == acc_second]
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
    output_file_path = data_folder + f"{file_name}.txt"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Open the file for writing (you can also use "a" to append if the file already exists)
    with open(output_file_path, "w") as file:
        # Write each string from the list to the file
        for string in list_outs:
            file.write(string + "\n")
    print(f'results saved to {output_file_path}')
    return list_outs

def recall_retrieval_evaluation(
        query_latents, 
        target_latents, 
        list_ks=[5, 10, 50, 100], 
        data_folder = "./internal_val_retrieval_results/",
        file_name='xray2ct',
        batch_size=1024):

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
            
            # find the top k indiices
            top_k_indices = find_top_k_indices(crosses, value)
            if i in top_k_indices:
                num_is_in += 1

            # this is the baseline performance on the random pairs.
            # for _ in range(len(dataloader)): # number of batches
            #     size = (512,)
            #     target_batch = torch.rand((batch_size, *size)).to('cuda')
            #     targets = torch.rand((batch_size, *size)).to('cuda')

            #     # Compute similarity in batch
            #     cross_batch = torch.matmul(target_batch, targets.T)
            #     crosses_rands.extend(cross_batch.cpu().tolist())

            # top_k_indices = find_top_k_indices(crosses_rands, value)
            # if i in top_k_indices:
            #     num_random += 1

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
    return list_texts

@hydra.main(
        version_base=None,
        config_path="/cluster/home/t135419uhn/CT-CLIP/configs",
        config_name="train")
def main(cfg: DictConfig):

    OmegaConf.resolve(cfg)

    if "LOCAL_RANK" in os.environ:
        # for ddp
        # passed by torchrun or torch.distributed.launch
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # for debugging
        local_rank = -1

    if local_rank < 1:
        print(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

    # seed_everything(1234)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # efficient performance optimization.

    cfg = convert_dictconfig_to_dict(cfg)

    # seed everything
    seed = 1024
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    run(cfg)


def run(cfg):
    torch.cuda.empty_cache()

    split = 'valid'
    embedding_directory = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/'
    # get the preprocessed image and text features
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
        file_name='ct2report_recall')

    print('evaluating report 2 ct in MAP')
    map_retrieval_evaluation(
        text_features,
        target_latents=image_features,
        predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        file_name='report2ct_map')

    # ct2ct
    print('evaluating ct 2 ct in MAP')
    map_retrieval_evaluation(
        image_features,
        target_latents=image_features,
        predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        file_name='ct2ct_map'
    )

    print('evaluating ct 2 report in MAP')
    map_retrieval_evaluation(
        image_features,
        target_latents=text_features,
        predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        file_name='ct2report_map')
    
    print('evaluating report 2 report in MAP')
    map_retrieval_evaluation(
        text_features,
        target_latents=text_features,
        predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv',
        file_name='report2report_map')

    print('Starting Xray related retrieval experiments')

    # windows wsl from local files
    tokenizer = BertTokenizer.from_pretrained(
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        do_lower_case=True,
        local_files_only=True)
    text_encoder = BertModel.from_pretrained(
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        local_files_only=True
        )

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
    #dim_image = 131072,

    # our retrival results: from cxr_clip model, from our pretrained xray encoder distilled from ct_clip
    ckpt_names = [
        # baseline pretrained model (not pretrained by us)
        'cxr_clip_swin', # xray encoder weights from cxr_clip
        'cxr_clip_resnet',
        'medclip_resnet',
        'medclip_vit',
        # 'gloria_densenet',
        # 'gloria_resnet',
        # our pretrained model
        'modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch',
        'modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch',
        # TODO: add the resnet one
    ]
    for ckpt_name in ckpt_names:

        #NOTE cfg is mainly for cxr_clip
        if 'cxr_clip' in ckpt_name: # can be either cxr_clip_swin or cxr_clip_resnet
            xray_model_type = ckpt_name #'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
            dim_xray = 768 if 'swin' in ckpt_name else 2048  # if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048
            pth_name = 'swin_cxr_xray_features.pth' if 'swin' in ckpt_name else 'resnet_cxr_xray_features.pth'
        elif ckpt_name == 'medclip_resnet':
            xray_model_type = ckpt_name
            dim_xray = 2048
            pth_name = 'swin_medclip_features.pth'

            # place this somewhere in the medclip code to remove the learnt fc connected layer at the end, just like cxr_clip: del self.resnet.fc
        elif ckpt_name == 'medclip_vit':
            xray_model_type = ckpt_name
            dim_xray = 768
            pth_name = 'resnet_medclip_features.pth'
        
        elif ckpt_name == 'gloria_densenet':
            xray_model_type = ckpt_name
            dim_xray = 1024 #TODO: double check this.
            pth_name = 'densenet_gloria_features.pth'

        elif ckpt_name == 'gloria_resnet':
            xray_model_type = ckpt_name
            dim_xray = 2048
            pth_name = 'resnet_gloria_features.pth'

        else:
            # our pretrained model
            xray_model_type = ckpt_name
            dim_xray = 768 if 'swin' in ckpt_name.lower() else 2048
            pth_name = f'{ckpt_name}_xray_features.pth'

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

        # NOTE: load the pretrained backbones
        # clip_xray, pth_name = load_model_weights(clip_xray, cfg, ckpt_name)

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


        print('evaluating xray 2 ct_volumes recall')
        recall_retrieval_evaluation(
            query_latents=[triple[-1] for triple in triplet_embeddings],
            target_latents=[triple[0].reshape(-1) for triple in triplet_embeddings],
            file_name=f'{ckpt_name}_synxray2ct_recall')
        print('evaluating ct_volumes 2 xray recall')
        recall_retrieval_evaluation(
            query_latents=[triple[0] for triple in triplet_embeddings],
            target_latents=[triple[-1].reshape(-1) for triple in triplet_embeddings],
            file_name=f'{ckpt_name}_ct2synxray_recall')



        print('evaluating xray 2 ct_reports recall')
        recall_retrieval_evaluation(
            query_latents=[triple[-1] for triple in triplet_embeddings],
            target_latents=[triple[1].reshape(-1) for triple in triplet_embeddings],
            file_name=f'{ckpt_name}_synxray2report_recall')
        print('evaluating ct_reports 2 xray recall')
        recall_retrieval_evaluation(
            query_latents=[triple[1] for triple in triplet_embeddings],
            target_latents=[triple[-1].reshape(-1) for triple in triplet_embeddings],
            file_name=f'{ckpt_name}_report2synxray_recall')



        print('evaluating xray 2 ct_volumes MAP')
        map_retrieval_evaluation(
            xray_features,
            target_latents=image_features,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{ckpt_name}_synxray2ct_map')
        print('evaluating ct_volumes 2 xray MAP')
        map_retrieval_evaluation(
            image_features,
            target_latents=xray_features,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{ckpt_name}_ct2synxray_map')



        print('evaluating xray 2 ct_reports MAP')
        map_retrieval_evaluation(
            xray_features,
            target_latents=text_features,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{ckpt_name}_synxray2report_map')
        print('evaluating ct_reports 2 xray MAP')
        map_retrieval_evaluation(
            text_features,
            target_latents=xray_features,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{ckpt_name}_report2synxray_map')



        # there is not symmetric retrieval and recall for this one.
        print('evaluating xray 2 xray MAP')
        map_retrieval_evaluation(
            xray_features,
            target_latents=xray_features,
            predicted_label_csv_path=f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
            file_name=f'{ckpt_name}_synxray2synxray_map')

if __name__ == '__main__':

    main()