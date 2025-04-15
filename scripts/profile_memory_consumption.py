import os
from cxr_clip_utils import convert_dictconfig_to_dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
# from torch_geometric import seed_everything
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIPwithXray
from CTCLIPTrainer import CTClipTrainer
import random
import numpy as np

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

    print("Batch Size:", cfg_dot.training_params.batch_size)
    print("Number of Workers:", cfg_dot.training_params.num_workers)
    print("Batch Style:", cfg_dot.training_params.batch_style)
    print("Train from Scratch:", cfg_dot.training_params.train_from_scratch)
    print("Epoch-Based Patience:", cfg_dot.training_params.epoch_based_patience)
    print("Iteration Evaluate Frequency:", cfg_dot.training_params.iteration_evaluate_frequency)
    print("Text Contrastive Learning Weight:", cfg_dot.training_params.text_cl_weight)
    print("CT Contrastive Learning Weight:", cfg_dot.training_params.ct_cl_weight)
    print("Learning Rate:", cfg_dot.training_params.learning_rate)
    print("Weight Decay:", cfg_dot.training_params.weight_decay)
    print("Epochs:", cfg_dot.training_params.epochs)
    print("Use Pretrained X-Ray Encoder:", cfg_dot.training_params.use_pretrained_xray_encoder)

    # convert the config file to dictionary
    cfg = convert_dictconfig_to_dict(cfg_dot)

    tokenizer = BertTokenizer.from_pretrained(
        '/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/CT_CLIP/BertTokenizer/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        do_lower_case=True,
        local_files_only=True
    )
    text_encoder = BertModel.from_pretrained(
        '/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/CT_CLIP/BertModel/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/f1cc2c6b7fac60f3724037746a129a5baf194dbc',
        local_files_only=True
    )

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

    if 'cxr_clip' in cfg_dot.training_params.training_pretrain_baseline: # can be either cxr_clip_swin or cxr_clip_resnet
        xray_model_type = cfg_dot.training_params.training_pretrain_baseline #'cxr_clip_swin' if cfg['model']['image_encoder']['model_type'] == 'swin' else 'cxr_clip_resnet'
        dim_xray = 768 if 'swin' in cfg_dot.training_params.training_pretrain_baseline else 2048  # if cfg['model']['image_encoder']['model_type'] == 'swin' else 2048
    elif cfg_dot.training_params.training_pretrain_baseline == 'medclip_resnet':
        xray_model_type = cfg_dot.training_params.training_pretrain_baseline
        dim_xray = 2048
        # place this somewhere in the medclip code to remove the learnt fc connected layer at the end, just like cxr_clip: del self.resnet.fc
    elif cfg_dot.training_params.training_pretrain_baseline == 'medclip_vit':
        xray_model_type = cfg_dot.training_params.training_pretrain_baseline
        dim_xray = 768
    elif cfg_dot.training_params.training_pretrain_baseline == 'gloria_densenet':
        xray_model_type = cfg_dot.training_params.training_pretrain_baseline
        dim_xray = 1024 #TODO: double check this.
    elif cfg_dot.training_params.training_pretrain_baseline == 'gloria_resnet':
        xray_model_type = cfg_dot.training_params.training_pretrain_baseline
        dim_xray = 2048
    else:
        xray_model_type = cfg_dot.training_params.training_pretrain_baseline
        dim_xray = 768 if 'swin' in cfg_dot.training_params.training_pretrain_baseline.lower() else 2048


    # for custom pretrained weight training
    latent_size = 512
    clip_xray = CTCLIPwithXray(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_text = 768,
        dim_image = 294912,
        xray_model_type = xray_model_type,
        dim_xray = dim_xray, # output size of the xray feature extractor
        dim_latent = latent_size, # latent size that match the CT vision encoder and the text encoder.
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False,
        cfg=cfg,
        auto_load_pretrained_weights=True if cfg_dot.training_params.use_pretrained_xray_encoder else False,
        freeze_xray_pretrained_weights=False, # need the xray encoder for training => no freeze parameters in xray encoder
        loss = cfg_dot.training_params.loss_function,
        projector_type=cfg_dot.training_params.projector_type,
    )
    # load the ct-clip pretrained weights
    clip_xray.load_ctclip('/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/models/CT-CLIP_v2.pt', freeze_weights=False)
    print('Finished loading the CTCLIP weights')

    # Dummy 3D input: [Batch, Channels, Depth, Height, Width]
    # note that 500 of width and height are the dimensions used for training the ct-clip
    # NOTE: change the slice, height, width dimension to observe the memory consumption
    dummy_input = torch.randn(4, 1, 240, 480, 480)  # Example: 1 sample, 1 channel, 64 slices of 128x128
    profile_memory(clip_xray.CTCLIP.visual_transformer, dummy_input)

def profile_memory(model, input_tensor, loss_fn=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Print model weight size
    total_params = sum(p.numel() for p in model.parameters())
    total_size_MB = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"Model parameters: {total_params:,} (~{total_size_MB:.2f} MB)")

    # Print input tensor size
    input_size_MB = input_tensor.numel() * input_tensor.element_size() / (1024**2)
    print(f"Input tensor size: {input_tensor.shape} (~{input_size_MB:.2f} MB)")

    # Clear cache and reset stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Warm-up (not timed)
    with torch.no_grad():
        _ = model(input_tensor, return_encoded_tokens=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    input_tensor.requires_grad = True

    # Forward pass
    torch.cuda.synchronize()
    start_forward_mem = torch.cuda.memory_allocated(device)
    output = model(input_tensor, return_encoded_tokens=True)
    torch.cuda.synchronize()
    end_forward_mem = torch.cuda.memory_allocated(device)

    if loss_fn is None:
        loss = output.sum()
    else:
        loss = loss_fn(output)

    # Backward pass with peak memory tracking
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()
    peak_mem_after_backward = torch.cuda.max_memory_allocated(device)

    forward_used = (end_forward_mem - start_forward_mem) / (1024**2)
    backward_used = (peak_mem_after_backward - end_forward_mem) / (1024**2)

    # print(f"Memory allocated at start of forward: {start_forward_mem / (1024**2):.2f} MB")
    # print(f"Memory allocated at end of forward: {end_forward_mem / (1024**2):.2f} MB")
    print(f"Memory used during forward pass: {forward_used:.2f} MB")
    print(f'Memory used during backward pass (include gradient information) {peak_mem_after_backward / (1024**2):.2f} MB')
    # print(f"Peak memory during backward pass (relative to end of forward): {backward_used:.2f} MB")

    return forward_used, backward_used


if __name__ == '__main__':
    main()