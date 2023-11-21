# reference: https://github.com/aimedical/clearml-dvis-scc/blob/main/Dockerfile

ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN distro=$(. /usr/lib/os-release; echo $ID$VERSION_ID | tr -d ".") \
 && arch=$(/usr/bin/arch) \
 && apt-key del 7fa2af80 \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/7fa2af80.pub

#
# 日本時間に設定する (ログを見やすくするため)
#
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev tzdata \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Tokyo

RUN conda clean --all

WORKDIR /workspace

#
# ClearML
#
RUN pip3 install --no-cache-dir \
    clearml \
    clearml-agent \
    boto3

#
# DVIS
#
RUN pip3 install --no-cache-dir detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
RUN pip3 install --no-cache-dir git+https://github.com/cocodataset/panopticapi.git
RUN pip3 install --no-cache-dir -U opencv-python

COPY docker/requirements.txt /workspace
RUN pip3 install -r requirements.txt

COPY docker/ops /workspace/ops
WORKDIR /workspace/ops
RUN FORCE_CUDA=1 sh make.sh
WORKDIR /workspace