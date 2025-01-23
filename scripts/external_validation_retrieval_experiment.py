"""
retrieval experiment for mimic
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
from zero_shot import MimicCTClipInference
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

    elif ckpt_name == 'cxr_clip_vit':
        assert cfg['model']['image_encoder']['model_type'] == 'swin' # make sure no conflict of expectation
        ckpt_file_name = 'swint_mcc'
        clip_xray.load_xray_encoder(
            '/cluster/home/t135419uhn/CT-CLIP/models/cxr_clip/{}.tar'.format(ckpt_file_name), # cxr-clip pretrained
            freeze_weights=True
        )
        pth_name = 'cxr_xray_swin_features.pth'
        print('Loaded weights from cxr_clip swin variant')
    
    elif ckpt_name == 'cxr_clip_resnet':
        assert cfg['model']['image_encoder']['name'] == 'resnet' # make sure no conflict of expectation
        ckpt_file_name = 'r50_mcc'
        clip_xray.load_xray_encoder(
            '/cluster/home/t135419uhn/CT-CLIP/models/cxr_clip/{}.tar'.format(ckpt_file_name), # cxr-clip pretrained
            freeze_weights=True
        )
        pth_name = 'cxr_xray_resnet_features.pth'
        print('Loaded weights from cxr_clip resnet variant')

    elif ckpt_name == 'medclip_resnet':
        pass
    
    elif ckpt_name == 'medclip_vit':
        pass

    elif ckpt_name == 'gloria_densenet':
        pass

    elif ckpt_name == 'gloria_resnet':
        pass

    else:
        # NOTE: our weights
        # ckpt_name='modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch'
        clip_xray.load_pretrained_ct_xray_clip(f'/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/{ckpt_name}.pt')
        pth_name = f'{ckpt_name}_xray_features.pth'
        print(f'Loaded weights from {ckpt_name}')

    return clip_xray, pth_name


def recall_retrieval_evaluation(
        query_latents, 
        target_latents, 
        list_ks=[5, 10, 50], 
        data_folder = "./retrieval_results2/",
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


def map_retrieval_evaluation(
        cfg_dot,
        query_latents, # dictionary of the xray latents
        target_latents, # xray or CT feature dictionary
        data_folder = "./retrieval_results2/",
        predicted_label_csv_path='path_to_valid_predicted_labels.csv',
        k_list=[1,5,10,50],
        batch_size=1024,
        file_name='xray2ct',
    ):
    assert 'mimic' in predicted_label_csv_path

    # convert the xray key as the accession to access the label later on.
    image_data_list = []
    accs = []
    for xray_file_key in tqdm.tqdm(query_latents.keys()):
        image_data_list.append(query_latents[xray_file_key]) # insert the embeddings
        accs.append(xray_file_key) #NOTE: # Use the filename without the extension as the accession number

    # Concatenate all loaded image data
    image_data = np.array(image_data_list)
    print(image_data.shape)

    # mainly for reading the file labels.
    df = pd.read_csv(predicted_label_csv_path)

    # Filter the image data based on the condition in the validation labels
    running_ratios_external = []
    image_data_for_second = []
    accs_for_second = []
    for target_key in tqdm.tqdm(target_latents.keys()):

        acc_second = target_key
        row_second = df[df['hadm_id'] == acc_second]
        num_path = np.sum(row_second.iloc[:, 1:].values[0]) # check if multilabel

        # if there are any labels (multihot or onehot) for this, save the embeddings and the file name NOTE: do we need this?
        if num_path != 0:
            target_latent = target_latents[target_key]
            image_data_for_second.append(target_latent)
            accs_for_second.append(acc_second)
        # else:
        #     print(f'this instance {target_key} is healthy {row_second.iloc[:, 1:].values[0].tolist()}')
    image_data_for_second = np.array(image_data_for_second) # one huge matrix
    print(image_data_for_second.shape)

    list_outs = []
    # Calculate the similarity for each image in the dataset
    for return_n in k_list:
        ratios_external = [] # take note for this one.
        for i in tqdm.tqdm(range(image_data.shape[0])):
            first = image_data[i] # get the embedding
            first = torch.tensor(first).to('cuda') # place it in the GPU for batch processing.
            acc_first = accs[i]
            row_first = df[df['hadm_id'] == acc_first]
            row_first = row_first.iloc[:, 1:].values[0]

            # Create a DataLoader for batching processing, with respect to each row_first
            dataset = TensorDataset(torch.tensor(image_data_for_second))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # NOTE: shuffle is False is mandatory

            crosses = []
            ratios_internal = []
            for batch in dataloader:
                second = batch[0].to('cuda')
                cross_batch = torch.matmul(first, second.T)
                crosses.extend(cross_batch.cpu().tolist())

            top_k_indices = find_top_k_indices(crosses, return_n)
            for index in top_k_indices:
                acc_second = accs_for_second[index]
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
    output_file_path = data_folder + f"{file_name}.txt"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Open the file for writing (you can also use "a" to append if the file already exists)
    with open(output_file_path, "w") as file:
        # Write each string from the list to the file
        for string in list_outs:
            file.write(string + "\n")
    print(list_outs)
    print(f'results saved to {output_file_path}')
    return list_outs


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


    # seed everything
    seed = 1024
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    run(cfg)


def run(cfg_dot):
    torch.cuda.empty_cache()

    print('Starting Xray related retrieval experiments')
    cfg = convert_dictconfig_to_dict(cfg_dot)

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

    #NOTE cfg is mainly for cxr_clip
    if cfg_dot.baseline_type == 'cxr_clip':
        xray_model_type = 'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
        dim_xray = 768 if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048
    elif cfg_dot.baseline_type == 'medclip_resnet':
        xray_model_type = cfg_dot.baseline_type
        dim_xray = 2048
        # place this somewhere in the medclip code to remove the learnt fc connected layer at the end, just like cxr_clip: del self.resnet.fc
    elif cfg_dot.baseline_type == 'medclip_vit':
        xray_model_type = cfg_dot.baseline_type
        dim_xray = 768
    elif cfg_dot.baseline_type == 'gloria_densenet':
        xray_model_type = cfg_dot.baseline_type
        dim_xray = 1024 #TODO: double check this.
    elif cfg_dot.baseline_type == 'gloria_resnet':
        xray_model_type = cfg_dot.baseline_type
        dim_xray = 2048

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
        baseline_type=cfg_dot.baseline_type # this dictate how the xray encoder is loaded to the model
    )

    # our retrival results: from cxr_clip model, from our pretrained xray encoder distilled from ct_clip
    ckpt_names = [
        'cxr_clip_vit', # xray encoder weights from cxr_clip
        #our pretrained model
        'modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch',
    ]
    for ckpt_name in ckpt_names:
        # NOTE: load the pretrained backbones
        clip_xray, _ = load_model_weights(clip_xray, cfg, ckpt_name)

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

        print('evaluating xray 2 ct_report MAP')
        map_retrieval_evaluation(
            cfg_dot,
            xray_features,
            target_latents=text_features,
            predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_external_valid_mimic_labels.csv',
            file_name=f'{ckpt_name}_mimic_xray2report_map')

        print('evaluating report 2 xray MAP')
        map_retrieval_evaluation(
            cfg_dot,
            text_features,
            target_latents=xray_features,
            predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_external_valid_mimic_labels.csv',
            file_name=f'{ckpt_name}_report2mimic_xray_map')

        # xray2xray retrieval evaluation with mean average precision metric
        print('evaluating xray 2 xray MAP')
        map_retrieval_evaluation(
            cfg_dot,
            xray_features,
            target_latents=xray_features,
            predicted_label_csv_path='/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_external_valid_mimic_labels.csv',
            file_name=f'{ckpt_name}_mimic_xray2mimic_xray_map')

        # # organize data into a list with index as a the text-image-xray correspondance and pair up xray-ct_image and xray-text
        triplet_embeddings = [('', text_features[key], xray_features[key]) for key in xray_features.keys()]

        print('evaluating report 2 xray recall')
        recall_retrieval_evaluation(
            query_latents=[triple[1] for triple in triplet_embeddings],
            target_latents=[triple[-1].reshape(-1) for triple in triplet_embeddings],
            file_name=f'{ckpt_name}_report2mimic_xray_recall')

        print('evaluating xray 2 ct reports recall')
        recall_retrieval_evaluation(
            query_latents=[triple[-1] for triple in triplet_embeddings],
            target_latents=[triple[1].reshape(-1) for triple in triplet_embeddings],
            file_name=f'{ckpt_name}_mimic_xray2report_recall')

if __name__ == '__main__':
    main()