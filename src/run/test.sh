#!/bin/bash
#$ -j yes
#$ -N inference
#$ -o /home/eliasse/colornet/logs/small_inference.log
#$ -l 'mem_free=16G,ram_free=16G,gpu=1,hostname=b1[1-9]*|c*
source /home/eliasse/anaconda3/etc/profile.d/conda.sh
conda activate dlproj
cd /home/eliasse/colornet/src/models
python -u inference.py --use-gpu --model-path=/home/eliasse/colornet/models/tensorboard_logs/models/0.01-0.49-2048_mymodel_8.pth

