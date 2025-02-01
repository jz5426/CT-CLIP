#!/bin/bash

#SBATCH -t 70:00:00
#SBATCH --mem=10G
#SBATCH -J vinbig_xray_preprocess
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --begin=now

source activate ctclip

python /cluster/home/t135419uhn/CT-CLIP/data_preprocess/preprocess_external_xray.py
