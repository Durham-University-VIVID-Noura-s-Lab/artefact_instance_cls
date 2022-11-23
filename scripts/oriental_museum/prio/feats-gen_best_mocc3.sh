#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name feats-gen_best_mocc3
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/feats-gen_best_mocc3.out
cd ../../..
source venv/bin/activate
python main.py --lr=-4.8 --optimiser=Adam --feats_gen \
    --model enetb6 \
    --test_ckpt_path ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_enetb6-epoch=74.ckpt" \
    --dataset=oriental-museum --device=0 --vt_bsz=100 --shuffle --metadata=instance --transforms=resize-by-model --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --epochs=100 --encoder_freeze=0 --bsz=32 --dset_seed=2667 --min_occ=3 --loss_weight_scaling
