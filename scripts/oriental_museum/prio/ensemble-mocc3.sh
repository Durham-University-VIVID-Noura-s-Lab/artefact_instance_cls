#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 8G
#SBATCH -p res-gpu-small
#SBATCH --job-name ensemble-mocc3
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/ensemble-mocc3.out
cd ../../..
source venv/bin/activate
python main.py --lr=-4.8 --optimiser=Adam --ensemble \
    --model enetb0 enetb1 enetb2 enetb3 enetb4 enetb5 enetb6 enetb7 inceptionv3 inceptionv4 resnet_rs_101 \
    --test_ckpt_path ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_enetb0-epoch=65.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_enetb1-epoch=93.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_enetb2-epoch=97.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_enetb3-epoch=92.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_enetb4-epoch=97.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_enetb5-epoch=98.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_enetb6-epoch=74.ckpt" ".results/bsz16-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_enetb7-epoch=92.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_inceptionv3-epoch=90.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_inceptionv4-epoch=99.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=3-seed2667_resnet_rs_101-epoch=93.ckpt" \
    --dataset=oriental-museum --device=0 --vt_bsz=100 --shuffle --metadata=instance --transforms=none --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --epochs=100 --encoder_freeze=0 --bsz=32 --dset_seed=2667 --min_occ=3 --loss_weight_scaling --wandb
