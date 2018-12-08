#!/bin/bash
#$ -j yes
#$ -N count
#$ -o /home/eliasse/colornet/src/stats/count.out
#$ -l 'mem_free=16G,ram_free=16G'
source /home/eliasse/anaconda3/etc/profile.d/conda.sh
conda activate dlproj
cd /home/eliasse/colornet/src/stats
python -u count.py 
