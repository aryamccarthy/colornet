#!/bin/bash
#$ -j yes
#$ -N inference
#$ -o /home/eliasse/colornet/logs/big_inference.log
#$ -l 'mem_free=16G,ram_free=16G,gpu=1,hostname=b1[1-9]*|c*
#$ -t 30-70:10
source /home/eliasse/anaconda3/etc/profile.d/conda.sh
conda activate dlproj
cd /home/eliasse/colornet/src/models
beta=$(bc <<<"scale=2;$SGE_TASK_ID/100")

python -u inference.py --use-gpu --model-path=/home/eliasse/colornet/models/tensorboard_logs/models/0.01-1-0$beta-2048_mymodel_10.pth
