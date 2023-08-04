#!/usr/bin/env bash

docker build docker/ -f docker/Dockerfile --network host --rm -t $USER/hubmap-2023
