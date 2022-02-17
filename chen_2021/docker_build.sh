#!/bin/bash
set -e
IMG="eval_chen2021:latest"
echo "image name: $IMG"
docker build . -t $IMG

