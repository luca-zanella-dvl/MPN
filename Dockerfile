# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.05-py3

# Install linux packages
RUN apt update && apt install -y tzdata ffmpeg libsm6 libxext6

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt wandb>=0.12.2
RUN pip install --no-cache mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install --no-cache -U torch torchvision numpy Pillow
# RUN pip install --no-cache torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
# COPY . /usr/src/app

# Set environment variables
ENV HOME=/usr/src/app
