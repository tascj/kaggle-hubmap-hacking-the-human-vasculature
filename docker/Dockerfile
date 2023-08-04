FROM nvcr.io/nvidia/pytorch:23.05-py3

ENV DEBIAN_FRONTEND=noninteractive


RUN python -m pip install -U pip
RUN pip install openmim
RUN mim install mmdet==3.0.0
RUN pip install mmpretrain==1.0.0rc8
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
