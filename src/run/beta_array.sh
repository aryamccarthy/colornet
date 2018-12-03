#!/bin/bash
#$ -j yes
#$ -N vary_beta 
#$ -o /home/eliasse/colornet/logs/beta_array.log
#$ -l 'mem_free=16G,ram_free=16G,gpu=1,hostname=b1[1-9]*|c*'
#$ -t 35-50:1
source /home/eliasse/anaconda3/etc/profile.d/conda.sh
conda activate dlproj
cd /home/eliasse/colornet/src/models
beta=$(bc <<<"scale=2;$SGE_TASK_ID/100")

python -u experiment.py --use-gpu=True --beta=$beta
