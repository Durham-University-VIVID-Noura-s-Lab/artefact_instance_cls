#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name om-i-resize-by-model-mocc-3-enetb0
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/om-i-resize-by-model-mocc-3-enetb0.out
cd ../../../..
source venv/bin/activate
python main.py --lr=-4.8 --optimiser=Adam --model=enetb0 --dataset=oriental-museum --device=0 --vt_bsz=100 --shuffle --metadata=instance --transforms=resize-by-model --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --epochs=100 --wandb --encoder_freeze=0 --bsz=32 --dset_seed=2667 --min_occ=3 --loss_weight_scaling \
    --conf_matrix --test_ckpt_path ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_enetb0-epoch=65.ckpt"
