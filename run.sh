#!/bin/bash
#$ -N train
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
source /groups/gcb50169/chen/nlp/bin/activate
module load python/3.6/3.6.12
module load cuda/11.1/11.1.1


python -u train.py
#qsub -g gcb50169 -l RESOURCE_TYPE=NUM_RESOURCE BATCH_FILE
