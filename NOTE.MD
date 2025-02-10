install the following

- conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

- follow the instruction from ct-clip repo but install the transformer (uncomment) in the transformer_maskgit and make sure comment the torch==2.0.1

pip uninstall attr
pip install attrs pydicom albumentations scikit-multilearn SimpleITK hydra-core

# for the bi-mamba environment
# note use cat /etc/os-release  to check the linux version for cuda toolkit installation
# note use uname -m to check the architecture of the linux machine

- or follow [here](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Rocky&target_version=9)
- install the toolkit system wide:
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run

- conda clean --all
- conda create -n bi-mamba python=3.10
- conda activate bi-mamba
<!-- - conda install nvidia/label/cuda-11.8.0::cuda-nvcc (the nvcc cuda version must matches the pytorch version, in this case 11.8) (must install this before any other) -->
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 （in the windows machine, i have 12.4 installed, as long as it matches the cuda version in the conda environment it should be fine.）
- pip install timm
- follow the instruction from https://github.com/hustvl/Vim and do the following steps
      - git clone https://github.com/RPIDIAL/BI-Mamba.git
      - go into the BI-Mamba/causal_conv1d
         - pip install -e .
      - go into the BI-Mamba/mamba-1p1p1
        - pip install -e .

<!-- download this repo https://github.com/state-spaces/mamba/tree/main and inside the repo, run pip install . -->
