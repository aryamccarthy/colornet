#!/bin/bash
#$ -j yes
#$ -N run_3 
#$ -o /home/eliasse/colornet/logs/run_3.log
#$ -l 'mem_free=16G,ram_free=16G,gpu=1,hostname=b1[1-9]*|c*
source /home/eliasse/anaconda3/etc/profile.d/conda.sh
conda activate dlproj
cd /home/eliasse/colornet/src/models
python -u experiment.py --use-gpu=True --epochs=800 --layers=3
