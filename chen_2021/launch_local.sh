#!/bin/bash
set -e
source ./docker_build.sh

# When running locally on some linux distros, must run ldconfig as root inside
# the container in order to enable CUDA. The container is launched as root,
# ldconfig is called and the eval script then runs as the user.
docker run \
  --gpus all \
  --device /dev/nvidia0  \
  --device /dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools \
  --device /dev/nvidiactl \
  -it -u root --entrypoint /bin/bash $IMG -c "ldconfig && runuser -u someuser -- bash evaluate.sh"
