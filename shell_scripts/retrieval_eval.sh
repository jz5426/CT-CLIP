#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J retrieval_2
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate ctclip
# # mimic
# python /cluster/home/t135419uhn/CT-CLIP/scripts/retrieval_evaluation.py retrieval_params.evaluation_dataset=mimic

# # radchest_ct_pure
python /cluster/home/t135419uhn/CT-CLIP/scripts/retrieval_evaluation.py retrieval_params.evaluation_dataset=radchest_ct_pure

# # # internal retrieval
# python /cluster/home/t135419uhn/CT-CLIP/scripts/retrieval_evaluation.py retrieval_params.evaluation_dataset=ct-rate

# # # radchest_ct
python /cluster/home/t135419uhn/CT-CLIP/scripts/retrieval_evaluation.py retrieval_params.evaluation_dataset=radchest_ct