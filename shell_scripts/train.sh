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
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.min_epochs=50 training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=51 training_params.use_pretrained_xray_encoder=true training_params.ct_cl_weight=1 training_params.text_cl_weight=0 training_params.loss_function=infoNCE training_params.projector_type=infoNCE
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.min_epochs=50 training_params.training_pretrain_baseline=cxr_clip_resnet training_params.epochs=51 training_params.use_pretrained_xray_encoder=true training_params.ct_cl_weight=1 training_params.text_cl_weight=0 training_params.loss_function=infoNCE training_params.projector_type=infoNCE
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.min_epochs=50 training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=51 training_params.use_pretrained_xray_encoder=true training_params.ct_cl_weight=0 training_params.text_cl_weight=1 training_params.loss_function=infoNCE training_params.projector_type=infoNCE
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.min_epochs=50 training_params.training_pretrain_baseline=cxr_clip_resnet training_params.epochs=51 training_params.use_pretrained_xray_encoder=true training_params.ct_cl_weight=0 training_params.text_cl_weight=1 training_params.loss_function=infoNCE training_params.projector_type=infoNCE

python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.min_epochs=50 training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=51 training_params.use_pretrained_xray_encoder=false training_params.ct_cl_weight=1 training_params.text_cl_weight=0 training_params.loss_function=infoNCE training_params.projector_type=infoNCE
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.min_epochs=50 training_params.training_pretrain_baseline=cxr_clip_resnet training_params.epochs=51 training_params.use_pretrained_xray_encoder=false training_params.ct_cl_weight=1 training_params.text_cl_weight=0 training_params.loss_function=infoNCE training_params.projector_type=infoNCE
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.min_epochs=50 training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=51 training_params.use_pretrained_xray_encoder=false training_params.ct_cl_weight=0 training_params.text_cl_weight=1 training_params.loss_function=infoNCE training_params.projector_type=infoNCE
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.min_epochs=50 training_params.training_pretrain_baseline=cxr_clip_resnet training_params.epochs=51 training_params.use_pretrained_xray_encoder=false training_params.ct_cl_weight=0 training_params.text_cl_weight=1 training_params.loss_function=infoNCE training_params.projector_type=infoNCE

#TODO: before start the training, modify the script so that it save at the 500 epochs instead of override the existing one. train longer the better.
## train with custom pretrained weights
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.epochs=500 training_params.use_pretrained_xray_encoder=false training_params.training_pretrain_baseline=cxr_clip_swin
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=instance training_params.epochs=500 training_params.use_pretrained_xray_encoder=false training_params.training_pretrain_baseline=cxr_clip_swin

# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.epochs=500 training_params.use_pretrained_xray_encoder=true training_params.training_pretrain_baseline=cxr_clip_swin
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=instance training_params.epochs=500 training_params.use_pretrained_xray_encoder=true training_params.training_pretrain_baseline=cxr_clip_swin


## run with siamese loss function with predictor and 3-layers projector (the projector and the loss functions are different)
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=52 training_params.use_pretrained_xray_encoder=false training_params.loss_function=siamese training_params.projector_type=siamese


# # instance pretrain is true infoNCE
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=instance training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=52 training_params.min_epochs=50 training_params.use_pretrained_xray_encoder=true training_params.loss_function=infoNCE training_params.projector_type=infoNCE
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=instance training_params.training_pretrain_baseline=cxr_clip_resnet training_params.epochs=52 training_params.min_epochs=50 training_params.use_pretrained_xray_encoder=true training_params.loss_function=infoNCE training_params.projector_type=infoNCE

# # instance pretrain is false infoNCE
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=instance training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=52 training_params.min_epochs=50 training_params.use_pretrained_xray_encoder=false training_params.loss_function=infoNCE training_params.projector_type=infoNCE
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=instance training_params.training_pretrain_baseline=cxr_clip_resnet training_params.epochs=52 training_params.min_epochs=50 training_params.use_pretrained_xray_encoder=false training_params.loss_function=infoNCE training_params.projector_type=infoNCE

# # experiment pretrain is true siamese
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=52 training_params.min_epochs=50 training_params.use_pretrained_xray_encoder=true training_params.loss_function=siamese training_params.projector_type=infoNCE
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=experiment training_params.training_pretrain_baseline=cxr_clip_resnet training_params.epochs=52 training_params.min_epochs=50 training_params.use_pretrained_xray_encoder=true training_params.loss_function=siamese training_params.projector_type=infoNCE

# # instance pretrain is true siamese
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=instance training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=52 training_params.min_epochs=50 training_params.use_pretrained_xray_encoder=true training_params.loss_function=siamese training_params.projector_type=infoNCE
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=instance training_params.training_pretrain_baseline=cxr_clip_resnet training_params.epochs=52 training_params.min_epochs=50 training_params.use_pretrained_xray_encoder=true training_params.loss_function=siamese training_params.projector_type=infoNCE

# # instance pretrain is false siamese
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=instance training_params.training_pretrain_baseline=cxr_clip_swin training_params.epochs=52 training_params.min_epochs=50 training_params.use_pretrained_xray_encoder=false training_params.loss_function=siamese training_params.projector_type=infoNCE
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_train.py training_params.batch_style=instance training_params.training_pretrain_baseline=cxr_clip_resnet training_params.epochs=52 training_params.min_epochs=50 training_params.use_pretrained_xray_encoder=false training_params.loss_function=siamese training_params.projector_type=infoNCE