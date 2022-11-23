#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name saliency_best_mocc6
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/saliency_best_mocc6.out
cd ../../..
source venv/bin/activate
python misc/saliency.py \
    --model enetb6 \
    --test_ckpt_path "../.results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_enetb6-epoch=97.ckpt" \
    --dataset=oriental-museum --device=0 --shuffle --metadata=instance --transforms=resize-by-model --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --encoder_freeze=0 --dset_seed=2667 --min_occ=6 --loss_weight_scaling --ensemble --method saliency
