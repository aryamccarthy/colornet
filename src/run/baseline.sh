#!/bin/bash
#$ -j yes
#$ -N lin_base
#$ -o /home/eliasse/colornet/logs/baseline.log
#$ -l 'mem_free=16G,ram_free=16G,gpu=1,hostname=b1[1-9]*|c*'

source /home/eliasse/anaconda3/etc/profile.d/conda.sh
conda activate dlproj
cd /home/eliasse/colornet/src/baseline
python -u linear.py --use-gpu=True --n-epochs=1
