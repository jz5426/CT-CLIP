"""
run the following script to extract xray features

TODO:
repeat the following and accumulate the stats:
    1. forward pass a synethic xray for each CT image in the validation set, to the pretrained xray encoder from ULIP style training
    2. with the embedding from xray encoder, compare it with the existing embeddings of the CT images based the image_embeddings.pth and find the top-k accuracy
        - note that since we forward pass the syntheic xray of examing ct image, we know the ground truth based on the dictionary id from image_embeddings.
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


def find_top_k_indices(values, k):
    # Check if the list has at least 50 values
    if len(values) < k:
        raise ValueError(f"The list must contain at least {k} values")

    # Use a combination of 'sorted' and 'enumerate' to sort the values while keeping track of indices
    sorted_values_with_indices = sorted(enumerate(values), key=lambda x: x[1], reverse=True)

    # Extract the indices of the top k values
    top_k_indices = [index for index, value in sorted_values_with_indices[:k]]

    return top_k_indices

def retrieval_evaluation(
        xray_latents, 
        target_latents, 
        list_ks=[5,10,50,100], 
        data_folder = "./retrieval_results/",
        file_name='xray2ct',
        batch_size=1000):

    xray_latents = np.array(xray_latents)
    target_latents = np.array(target_latents) # to be retrieved from

    list_texts = []
    for value in tqdm.tqdm(list_ks):
        num_is_in, num_random = 0, 0

        # for each xray => the goal is to retrieve the correct target
        for i in tqdm.tqdm(range(xray_latents.shape[0])):
            crosses, crosses_rands = [], []
            xray = torch.tensor(xray_latents[i]).to('cuda')

            # Create a DataLoader for batching
            dataset = TensorDataset(torch.tensor(target_latents))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            #TODO: normalize the feature before testing?

            # find the similarity between the xray and the target embeddings
            for batch in dataloader:
                targets = batch[0].to('cuda')
                
                # Compute similarity in batch
                cross_batch = torch.matmul(xray, targets.T)
                crosses.extend(cross_batch.cpu().tolist())
            
            # find the top k indiices
            top_k_indices = find_top_k_indices(crosses, value)
            if i in top_k_indices:
                num_is_in += 1

            # this is the baseline performance on the random pairs.
            for _ in range(len(dataloader)): # number of batches
                size = (512,)
                target_batch = torch.rand((batch_size, *size)).to('cuda')
                targets = torch.rand((batch_size, *size)).to('cuda')

                # Compute similarity in batch
                cross_batch = torch.matmul(target_batch, targets.T)
                crosses_rands.extend(cross_batch.cpu().tolist())

            top_k_indices = find_top_k_indices(crosses_rands, value)
            if i in top_k_indices:
                num_random += 1

        clip = num_is_in / target_latents.shape[0]
        rand = num_random / target_latents.shape[0]
        write_str = f"K={value}, clip = {clip}, rand= {rand}"
        # write_str = f"K={value}, clip = {clip}, rand=undefined"

        list_texts.append(write_str)

    # output_file_path = data_folder + f"internal_accessions_t2i_{list_ks[0]}.txt"
    output_file_path = data_folder + f"{file_name}.txt"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Open the file for writing (you can also use "a" to append if the file already exists)
    with open(output_file_path, "w") as file:
        # Write each string from the list to the file
        for string in list_texts:
            file.write(string + "\n")

    return list_texts


@hydra.main(
        version_base=None,
        # config_path="C:\\Users\\MaxYo\\OneDrive\\Desktop\\MBP\\chris\\CT-CLIP\\configs",
        # config_path="/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/configs",
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
    #NOTE: you need to use the follownig command to copy and past to the location cp -rL /path/to/source_directory /path/to/destination_directory 
        # the copied files in the destination folder will behave like regular files and directories. You can copy and paste them as usual using a file manager

    # windows wsl from download    
    # tokenizer = BertTokenizer.from_pretrained(
    #     'microsoft/BiomedVLP-CXR-BERT-specialized',
    #     do_lower_case=True,
    #     cache_dir='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/predownloaded_models/BertTokenizer')
    # text_encoder = BertModel.from_pretrained(
    #     'microsoft/BiomedVLP-CXR-BERT-specialized',
    #     cache_dir='/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/predownloaded_models/BertModel'
    #     )

    # windows wsl from local files
    tokenizer = BertTokenizer.from_pretrained(
        # '/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/predownloaded_models/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        do_lower_case=True,
        local_files_only=True)
    text_encoder = BertModel.from_pretrained(
        # '/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        local_files_only=True
        )

    # uhn cluster from local filesc
    #TODO: 
        # 1. copy the downloaded huggingface model in G:\Chris\CT-CLIP\predownloaded_models (shield external drive) to the CT-CLIP
        # 2. for the image_encoder section of the yaml file (such as clip_Swin_clincial), replace the directory to the correct one
    # tokenizer = BertTokenizer.from_pretrained(
    #     '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
    #     do_lower_case=True,
    #     local_files_only=True
    # )
    # text_encoder = BertModel.from_pretrained(
    #     '/cluster/home/t135419uhn/CT-CLIP/predownloaded_models/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
    #     local_files_only=True
    # )

    print("---------")
    print(tokenizer.pad_token_id)
    print(tokenizer.mask_token_id)
    print("-----------")


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

    clip_xray = CTCLIPwithXray(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        # tokenizer=tokenizer,
        dim_text = 768,
        dim_image = 294912,
        xray_model_type = 'swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'resnet',
        dim_xray = 768 if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False,
        cfg=cfg
    )

    # NOTE: load the pretrained backbones
    ckpt_name = 'r50_mcc.tar' if cfg['model']['image_encoder']['name'] == 'resnet' else 'swint_mcc.tar'

    # NOTE: cxr-clip pretrained weights
    # clip_xray.load_xray_encoder(
    #     '/cluster/home/t135419uhn/CT-CLIP/models/cxr_clip/{}'.format(ckpt_name), # cxr-clip pretrained
    #     freeze_weights=True
    # )
    # pth_name = 'cxr_xray_features.pth'

    # NOTE: our weights
    ckp_name = 'CTClip.lowest_val_cl_loss_during_iterations'
    clip_xray.load_pretrained_ct_xray_clip(f'/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/{ckp_name}.pt')
    pth_name = f'{ckp_name}_xray_features.pth'

    # check the trainable parameters
    xray_encoder_trainable = sum(p.numel() for p in clip_xray.xray_encoder.parameters() if p.requires_grad)
    ct_clip_trainable = sum(p.numel() for p in clip_xray.CTCLIP.parameters() if p.requires_grad)
    # assert(xray_encoder_trainable == 0)
    # assert(ct_clip_trainable == 0)

    split = 'valid'
    retrival_evaluator = CTClipInference(
        clip_xray,
        cfg=cfg,
        tokenizer=tokenizer,
        # data_folder = "/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/valid_preprocessed_xray_mha",
        data_folder= f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/{split}_preprocessed_xray_mha',
        # NOTE: the embedding paths are MANDATORY for the dataloader to work. RUN THIS SCRIPT MAINLY AFTER THE CTCLIP EMBEDDINGS ARE EXTRACTED.
        img_embedding_paths = {
            # f'{split}': f'/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/features_embeddings/{split}/image_features.pth'
            f'{split}': f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/{split}/image_features.pth'
        },
        text_embedding_paths = {
            # f'{split}': f'/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/features_embeddings/{split}/text_features.pth'
            f'{split}': f'/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/{split}/text_features.pth'
        },
        # reports_file = f'/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/radiology_text_reports/{split}_reports.csv',
        reports_file = f'/cluster/home/t135419uhn/CT-CLIP/dataset/radiology_text_reports/{split}_reports.csv',
        # labels = f"/mnt/c/Users/MaxYo/OneDrive/Desktop/MBP/Chris/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv",
        labels = f'/cluster/home/t135419uhn/CT-CLIP/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_{split}_predicted_labels.csv',
        results_folder="./inference_zeroshot_retrieval",
        batch_size = 128,
        num_train_steps = -1, # placeholder
        num_workers = 10, # with the preprocess data as .pt file, the preprocessing should be fast, 1 is sufficient.
        feature_extraction_mode = True # might be optional
    )  

    # embedding_directory = '/mnt/f/Chris/CT-RATE-FINAL/processed_dataset/features_embeddings_correct'
    embedding_directory = '/cluster/projects/mcintoshgroup/publicData/CT-RATE/processed_dataset/features_embeddings/'

    xray_features = retrival_evaluator.xray_feature_extraction(embedding_directory, pth_name=pth_name)
    
    # get the image and text features
    saving_path = os.path.join(embedding_directory, split)
    img_feature_path = os.path.join(saving_path, 'image_features.pth')
    text_feature_path = os.path.join(saving_path, 'text_features.pth')
    image_features, text_features = None, None
    if os.path.exists(img_feature_path):
        image_features = torch.load(img_feature_path)
    if os.path.exists(text_feature_path):
        text_features = torch.load(text_feature_path)

    # make sure all three dictionary contains the same set of keys
    assert(image_features.keys() == text_features.keys() == xray_features.keys())
    
    # organize data into a list with index as a the text-image-xray correspondance and pair up xray-ct_image and xray-text
    triplet_embeddings = [(image_features[key], text_features[key], xray_features[key]) for key in xray_features.keys()]

    # xray2image retrival evaluation
    retrieval_evaluation(
        xray_latents=[triple[-1] for triple in triplet_embeddings],
        target_latents=[triple[0].reshape(-1) for triple in triplet_embeddings],
        file_name='synxray2ct.txt')

    # xray2report retrival evaluation
    retrieval_evaluation(
        xray_latents=[triple[-1] for triple in triplet_embeddings],
        target_latents=[triple[1].reshape(-1) for triple in triplet_embeddings],
        file_name='synxray2report.txt')

if __name__ == '__main__':
    main()