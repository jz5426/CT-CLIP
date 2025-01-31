#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J ctrate_xray_feat_extraction
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate ctclip

# train with custom pretrained weights

python /cluster/home/t135419uhn/CT-CLIP/scripts/ctrate_xray_feature_caching.py xray_feature_caching_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch
python /cluster/home/t135419uhn/CT-CLIP/scripts/ctrate_xray_feature_caching.py xray_feature_caching_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch
python /cluster/home/t135419uhn/CT-CLIP/scripts/ctrate_xray_feature_caching.py xray_feature_caching_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch
python /cluster/home/t135419uhn/CT-CLIP/scripts/ctrate_xray_feature_caching.py xray_feature_caching_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch
python /cluster/home/t135419uhn/CT-CLIP/scripts/ctrate_xray_feature_caching.py xray_feature_caching_params.baseline_type=cxr_clip_swin
python /cluster/home/t135419uhn/CT-CLIP/scripts/ctrate_xray_feature_caching.py xray_feature_caching_params.baseline_type=cxr_clip_resnet
python /cluster/home/t135419uhn/CT-CLIP/scripts/ctrate_xray_feature_caching.py xray_feature_caching_params.baseline_type=medclip_vit
python /cluster/home/t135419uhn/CT-CLIP/scripts/ctrate_xray_feature_caching.py xray_feature_caching_params.baseline_type=medclip_resnet
python /cluster/home/t135419uhn/CT-CLIP/scripts/ctrate_xray_feature_caching.py xray_feature_caching_params.baseline_type=gloria_densenet
python /cluster/home/t135419uhn/CT-CLIP/scripts/ctrate_xray_feature_caching.py xray_feature_caching_params.baseline_type=gloria_resnet
