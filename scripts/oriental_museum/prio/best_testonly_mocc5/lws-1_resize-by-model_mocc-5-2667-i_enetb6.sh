#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name om-i-resize-by-model-mocc-5-enetb6
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/om-i-resize-by-model-mocc-5-enetb6.out
cd ../../../..
source venv/bin/activate
python main.py --lr=-4.8 --optimiser=Adam --model=enetb6 --dataset=oriental-museum --device=0 --vt_bsz=8 --shuffle --metadata=instance --transforms=resize-by-model --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --epochs=100 --wandb --encoder_freeze=0 --bsz=4 --dset_seed=2667 --min_occ=5 --loss_weight_scaling \
    --test_ckpt_path ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=5-seed2667_enetb6-epoch=47.ckpt"
