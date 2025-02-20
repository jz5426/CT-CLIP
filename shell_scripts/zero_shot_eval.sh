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

# radchest ct final_labels_clean.csv
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=ct_clip zero_shot_params.test_bed=radchest_ct_internal # TODO:
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=radchest_ct_internal
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=radchest_ct_internal
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=radchest_ct_internal
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=radchest_ct_internal
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_swin_m zero_shot_params.test_bed=radchest_ct_internal
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_resnet_m zero_shot_params.test_bed=radchest_ct_internal
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_vit zero_shot_params.test_bed=radchest_ct_internal
python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_resnet zero_shot_params.test_bed=radchest_ct_internal


# radchest all disease ct only
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=ct_clip zero_shot_params.test_bed=radchest_all_disease_ct_only_internal # TODO:
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=radchest_all_disease_ct_only_internal
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=radchest_all_disease_ct_only_internal
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=radchest_all_disease_ct_only_internal
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=radchest_all_disease_ct_only_internal
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_swin_m zero_shot_params.test_bed=radchest_all_disease_ct_only_internal
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_resnet_m zero_shot_params.test_bed=radchest_all_disease_ct_only_internal
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_vit zero_shot_params.test_bed=radchest_all_disease_ct_only_internal
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_resnet zero_shot_params.test_bed=radchest_all_disease_ct_only_internal

# radchest ct pure
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=ct_clip zero_shot_params.test_bed=radchest_ct_pure # TODO:
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=radchest_ct_pure
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=radchest_ct_pure
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=radchest_ct_pure
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=radchest_ct_pure
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_swin_m zero_shot_params.test_bed=radchest_ct_pure
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_resnet_m zero_shot_params.test_bed=radchest_ct_pure
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_vit zero_shot_params.test_bed=radchest_ct_pure
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_resnet zero_shot_params.test_bed=radchest_ct_pure


# # internal eval
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=ct_clip zero_shot_params.test_bed=ct-rate
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=ct-rate
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=ct-rate
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=ct-rate
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=ct-rate
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_swin_m zero_shot_params.test_bed=ct-rate
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_resnet_m zero_shot_params.test_bed=ct-rate
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_vit zero_shot_params.test_bed=ct-rate
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_resnet zero_shot_params.test_bed=ct-rate

# # mimic eval
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=mimic
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Swin__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=mimic
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_True_50_epoch zero_shot_params.test_bed=mimic
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=modeltype_Resnet__batchstyle_experiment__bs_360__lr_5e-05__wd_0.0001__textcl_1.0__ctcl_1.0__pretrained_False_50_epoch zero_shot_params.test_bed=mimic
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_swin_m zero_shot_params.test_bed=mimic
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=cxr_clip_resnet_m zero_shot_params.test_bed=mimic
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_vit zero_shot_params.test_bed=mimic
# python /cluster/home/t135419uhn/CT-CLIP/scripts/run_xray_zero_shot.py zero_shot_params.baseline_type=medclip_resnet zero_shot_params.test_bed=mimic