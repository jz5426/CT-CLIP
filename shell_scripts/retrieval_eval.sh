#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 40:00:00
#SBATCH --mem=40G
#SBATCH -J ctrate_retrieval
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate ctclip

# # ablation study
python /cluster/home/t135419uhn/CT-CLIP/scripts/retrieval_evaluation.py retrieval_params.evaluation_dataset=ct-rate base.is_ablation_study=true

# # mimic
# python /cluster/home/t135419uhn/CT-CLIP/scripts/retrieval_evaluation.py retrieval_params.evaluation_dataset=mimic

# # radchest_ct_pure
# python /cluster/home/t135419uhn/CT-CLIP/scripts/retrieval_evaluation.py retrieval_params.evaluation_dataset=radchest_ct_pure

# # # radchest_ct
# python /cluster/home/t135419uhn/CT-CLIP/scripts/retrieval_evaluation.py retrieval_params.evaluation_dataset=radchest_ct