#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --qos short
#SBATCH --job-name om-i-resize-by-model-mocc-5-inceptionv3
#SBATCH --mem 8G
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/om-i-resize-by-model-mocc-5-inceptionv3.out
cd ../../../..
source venv/bin/activate
python main.py --lr=-4.8 --optimiser=Adam --model=inceptionv3 --dataset=oriental-museum --device=0 --vt_bsz=100 --shuffle --metadata=instance --transforms=resize-by-model --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --epochs=100 --wandb --encoder_freeze=0 --bsz=32 --dset_seed=2667 --min_occ=5 --loss_weight_scaling
