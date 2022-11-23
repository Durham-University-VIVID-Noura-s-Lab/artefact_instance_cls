#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 8G
#SBATCH -p res-gpu-small
#SBATCH --job-name ensemble-mocc6
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/ensemble-mocc6.out
cd ../../..
source venv/bin/activate
python main.py --lr=-4.8 --optimiser=Adam --ensemble \
    --model enetb0 enetb1 enetb2 enetb3 enetb4 enetb5 enetb6 enetb7 inceptionv3 inceptionv4 resnet_rs_101 \
    --test_ckpt_path ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_enetb0-epoch=94.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_enetb1-epoch=61.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_enetb2-epoch=80.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_enetb3-epoch=85.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_enetb4-epoch=96.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_enetb5-epoch=91.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_enetb6-epoch=97.ckpt" ".results/bsz16-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_enetb7-epoch=70.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_inceptionv3-epoch=99.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_inceptionv4-epoch=99.ckpt" ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_resnet_rs_101-epoch=97.ckpt" \
    --dataset=oriental-museum --device=0 --vt_bsz=100 --shuffle --metadata=instance --transforms=none --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --epochs=100 --encoder_freeze=0 --bsz=32 --dset_seed=2667 --min_occ=6 --loss_weight_scaling --wandb
