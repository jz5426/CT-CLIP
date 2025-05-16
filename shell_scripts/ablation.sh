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
# iterate through different set of weighting parameters
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.min_epochs=50 training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=51 training_params.use_pretrained_xray_encoder=true training_params.ct_cl_weight=1 training_params.text_cl_weight=0 training_params.loss_function=infoNCE training_params.projector_type=infoNCE
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.min_epochs=50 training_params.training_pretrain_baseline=cxr_clip_resnet training_params.epochs=51 training_params.use_pretrained_xray_encoder=true training_params.ct_cl_weight=1 training_params.text_cl_weight=0 training_params.loss_function=infoNCE training_params.projector_type=infoNCE
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.min_epochs=50 training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=51 training_params.use_pretrained_xray_encoder=true training_params.ct_cl_weight=0 training_params.text_cl_weight=1 training_params.loss_function=infoNCE training_params.projector_type=infoNCE
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.min_epochs=50 training_params.training_pretrain_baseline=cxr_clip_resnet training_params.epochs=51 training_params.use_pretrained_xray_encoder=true training_params.ct_cl_weight=0 training_params.text_cl_weight=1 training_params.loss_function=infoNCE training_params.projector_type=infoNCE
#TODO: before start the training, modify the script so that it save at the 500 epochs instead of override the existing one. train longer the better.
## train with custom pretrained weights
