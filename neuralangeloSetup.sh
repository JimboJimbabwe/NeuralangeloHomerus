#!/bin/bash

# Install CUDA and add graphics drivers PPA
sudo apt install -y cuda
sudo add-apt-repository -y ppa:graphics-drivers/ppa

# Create miniconda3 directory
mkdir -p ~/miniconda3

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# Install Miniconda
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# Remove Miniconda installer
rm -rf ~/miniconda3/miniconda.sh

# Initialize Conda for bash and zsh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

# Update packages
sudo apt-get update -y

# Install NVIDIA CUDA Toolkit
sudo apt install -y nvidia-cuda-toolkit

# Install FFmpeg and COLMAP
sudo apt install -y ffmpeg colmap

# Install Python packages
pip install -y addict
pip install -y termcolor
pip install -y matplotlib
pip install -y wandb
pip install -y tinycudann
pip install -y PIL

# Install build essentials and Git
sudo apt update -y && sudo apt upgrade -y && sudo apt-get install -y build-essential git g++

# Clone the neuralangelo repository
git clone https://github.com/NVlabs/neuralangelo.git

# Change permissions of the neuralangelo folder
sudo chmod -R 777 neuralangelo

# Create and activate the neuralangelo Conda environment
conda env create -y --file neuralangelo/neuralangelo.yaml
conda activate neuralangelo
conda install -c conda-forge libpng
conda install -c pytorch torchvision
pip install -r requirements.txt
cd neuralangelo
#establish a python path

export PYTHONPATH="$(pwd):$PYTHONPATH"
source  ~/.bashrc
conda activate neuralangelo


# Install additional packages (just in case)
pip install -y addict termcolor matplotlib wandb torch tinycudann

