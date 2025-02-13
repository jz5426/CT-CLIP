install the following

- conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

- follow the instruction from ct-clip repo but install the transformer (uncomment) in the transformer_maskgit and make sure comment the torch==2.0.1

pip uninstall attr
pip install attrs pydicom albumentations scikit-multilearn SimpleITK hydra-core

# for the bi-mamba environment
# note use cat /etc/os-release  to check the linux version for cuda toolkit installation
# note use uname -m to check the architecture of the linux machine

# if there is a error on there is an error on the casual_1d

- after installing the cuda toolkit for version 11.8, need to run the following.
  echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
  source ~/.bashrc



# if encounter the Mamba.__init__() got an unexpected keyword argument 'if_devide_out' error:
  - replace if_devide_out with if_divide_out across all the files


## Detail instruction on environment set with bi-mamba and ct-clip

# in summary

- install bimamba environment follow the instruction from Vim
  - refer the following cite: https://github.com/hustvl/Vim/issues/41 and https://www.a2p.it/machine-learning/pretrained-vision-mamba-minimal-example/ 
  - the key thing is download the Vim repo
    - change the causal_1d pacckage version to 1.1.0 (IMPORTANT) in the transformer_maskgit 
    - no need to install the causal_conv1d package
    - follow the rest of the Vim installation instruction
  - after installation of the bimamba environment, start to install the ct-clip environment
- install the ct-clip environment, might be except the pytorch version.
  - follow the instruction from ct-clip repo but install the transformer (uncomment) in the transformer_maskgit and make sure comment the torch==2.0.1
  pip uninstall attr
  pip install attrs pydicom albumentations scikit-multilearn SimpleITK hydra-core
