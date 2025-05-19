#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J ablation_Study
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate ctclip

##  here you put the python command line to run the code for training or hyperparameter search

# use_pretrained_xray_encoder = true
gammas=(0.1 0.2 0.4 0.8 1)
betas=(0.1 0.2 0.4 0.8 1)

# Iterate over all combinations of gamma and beta, initialized with pretrained weights
# for gamma in "${gammas[@]}"; do
#     for beta in "${betas[@]}"; do
#         echo "Running with gamma=${gamma} and beta=${beta}"
        
#         python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py \
#             training_params.batch_style=experiment \
#             training_params.min_epochs=50 \
#             training_params.training_pretrain_baseline=cxr_clip_swin \
#             training_params.epochs=51 \
#             training_params.use_pretrained_xray_encoder=true \
#             training_params.ct_cl_weight=${gamma} \
#             training_params.text_cl_weight=${beta} \
#             training_params.checkpoint_saving_directory='/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/Ablation' \
#             training_params.loss_function=infoNCE \
#             training_params.projector_type=infoNCE
#     done
# done

# Iterate over all combinations of gamma and beta, initialized with pretrained weights
for gamma in "${gammas[@]}"; do
    for beta in "${betas[@]}"; do
        echo "Running with gamma=${gamma} and beta=${beta}"
        
        python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py \
            training_params.batch_style=experiment \
            training_params.min_epochs=50 \
            training_params.training_pretrain_baseline=cxr_clip_swin \
            training_params.epochs=51 \
            training_params.use_pretrained_xray_encoder=false \
            training_params.ct_cl_weight=${gamma} \
            training_params.text_cl_weight=${beta} \
            training_params.checkpoint_saving_directory='/cluster/projects/mcintoshgroup/CT-RATE-CHECKPOINTS/Ablation' \
            training_params.loss_function=infoNCE \
            training_params.projector_type=infoNCE
    done
done
