#!/usr/bin/env bash

GPU_ID=$1

docker run --rm -it --gpus '"device='$GPU_ID'"' --ipc=host \
    -v $CODE_DIR:/workspace \
    -v $DATA_DIR:/data \
    -v /home/$USER/.cache/torch:/root/.cache/torch \
    -v /home/$USER/.cache/huggingface:/root/.cache/huggingface \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    $USER/hubmap-2023

