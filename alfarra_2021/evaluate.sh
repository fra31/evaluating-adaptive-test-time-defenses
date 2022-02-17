#!/bin/bash

# Copy required files to a working dir
cp evaluate.py /tmp/
cd /tmp

# Clone repo and install some dependencies
git clone https://github.com/MotasemAlfarra/Combating-Adversaries-with-Anti-Adversaries
git clone https://github.com/uclaml/RayS
sed -i "s/_, term_width = os.popen('stty size', 'r').read().split()/term_width=80/g" RayS/pgbar.py     # Fix terminal width in pgbar code

gdown --id 1sSjh4i2imdoprw_JcPj2cZzrJm0RIRI6
mkdir weights
mv RST-AWP_cifar10_linf_wrn28-10.pt weights/newmodel1_RST-AWP_cifar10_linf_wrn28-10.pt

python evaluate.py
