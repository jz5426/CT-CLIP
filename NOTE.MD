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
- install pytorch if not like in CT-CLIP
- conda install nvidia::cuda-nvcc or conda install nvidia/label/cuda-11.8.0::cuda-nvcc
- pip install timm
- follow the instruction from https://github.com/hustvl/Vim to install 
      - pip install -e causal_conv1d>=1.1.0
      - pip install -e mamba-1p1p1 (this is the mamba_ssm Mamba with bimamba feature)
<!-- download this repo https://github.com/state-spaces/mamba/tree/main and inside the repo, run pip install . -->
