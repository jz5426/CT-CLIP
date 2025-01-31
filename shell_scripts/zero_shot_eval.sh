#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J zero_shot_eval
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate ctclip
# the run order is the same as the table order in notion but missing gloria

# internal eval
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=ct_clip zero_shot_params.test_bed=internal_ct_val
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=internal_ct_val
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=internal_ct_val
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=internal_ct_val
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=internal_ct_val

python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_swin zero_shot_params.test_bed=internal_ct_val
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_resnet zero_shot_params.test_bed=internal_ct_val

python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_vit zero_shot_params.test_bed=internal_ct_val
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_resnet zero_shot_params.test_bed=internal_ct_val

# mimic eval
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=mimic_ct
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=mimic_ct
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=mimic_ct
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=mimic_ct

python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_swin zero_shot_params.test_bed=mimic_ct
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_resnet zero_shot_params.test_bed=mimic_ct

python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_vit zero_shot_params.test_bed=mimic_ct
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_resnet zero_shot_params.test_bed=mimic_ct