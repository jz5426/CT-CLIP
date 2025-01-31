#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J medclip_pretrain
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

## does not work since resnet does not have mechanism to load imagenet pretrained weights
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.epochs=202 training_params.use_pretrained_xray_encoder=false 


# train with custom pretrained weights
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.epochs=101 training_params.use_pretrained_xray_encoder=true training_params.training_pretrain_baseline=medclip_vit
