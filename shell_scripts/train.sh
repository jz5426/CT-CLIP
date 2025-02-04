#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J train
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate ctclip

# here you put the python command line to run the code for training or hyperparameter search
# then run the sbatch train.sh for 200 epochs and saving the intermediate 50, 100, 150, 200 checkpoints
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.epochs=202 training_params.use_pretrained_xray_encoder=true
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.epochs=202 training_params.use_pretrained_xray_encoder=false


#TODO: before start the training, modify the script so that it save at the 500 epochs instead of override the existing one. train longer the better.
## train with custom pretrained weights
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.epochs=500 training_params.use_pretrained_xray_encoder=false training_params.training_pretrain_baseline=cxr_clip_swin
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=instance training_params.epochs=500 training_params.use_pretrained_xray_encoder=false training_params.training_pretrain_baseline=cxr_clip_swin

# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.epochs=500 training_params.use_pretrained_xray_encoder=true training_params.training_pretrain_baseline=cxr_clip_swin
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=instance training_params.epochs=500 training_params.use_pretrained_xray_encoder=true training_params.training_pretrain_baseline=cxr_clip_swin


## run with siamese loss function with predictor and 3-layers projector (the projector and the loss functions are different)
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=52 training_params.use_pretrained_xray_encoder=false training_params.loss_function=siamese training_params.projector_type=siamese

## run with siamese loss function with no predictor and single layer projector (only the loss function is different)
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=52 training_params.use_pretrained_xray_encoder=false training_params.loss_function=siamese training_params.projector_type=infoNCE
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.training_pretrain_baseline=cxr_clip_resnet training_params.epochs=52 training_params.use_pretrained_xray_encoder=false training_params.loss_function=siamese training_params.projector_type=infoNCE
