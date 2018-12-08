#!/bin/bash
#$ -j yes
#$ -N cpu_run_exp 
#$ -o /home/eliasse/colornet/logs/cpu_run.log
#$ -l 'mem_free=16G,ram_free=16G'
source /home/eliasse/anaconda3/etc/profile.d/conda.sh
conda activate dlproj
cd /home/eliasse/colornet/src/models
python -u experiment.py 
