install the following

- conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

- follow the instruction from ct-clip repo but install the transformer (uncomment) in the transformer_maskgit and make sure comment the torch==2.0.1

pip uninstall attr
pip install attrs
pip install pydicom
pip install albumentations 
pip install scikit-multilearn
pip install SimpleITK
pip install hydra-core

# for the bi-mamba environment
# note use cat /etc/os-release  to check the linux version for cuda toolkit installation

conda clean --all
- conda create -n bi-mamba python=3.10
- conda activate bi-mamba
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- conda install nvidia::cuda-nvcc or conda install nvidia/label/cuda-11.8.0::cuda-nvcc (the nvcc cuda version must matches the pytorch version, in this case 11.8)
- pip install timm
- follow the instruction from https://github.com/hustvl/Vim and do the following steps
      - git clone https://github.com/hustvl/Vim
      - pip install -e causal_conv1d>=1.1.0
      - pip install -e mamba-1p1p1 (this is the mamba_ssm Mamba with bimamba feature)
      or go into the BI-Mamba/causal_conv1d
      - pip install -e .
      or go into the BI-Mamba/mamba-1p1p1
      - pip install -e .

<!-- download this repo https://github.com/state-spaces/mamba/tree/main and inside the repo, run pip install . -->
