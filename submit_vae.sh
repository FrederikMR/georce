#!/bin/sh
#BSUB -q gpuv100
#BSUB -J svhn
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=10GB]"
#BSUB -u fmry@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o sendmeemail/error_%J.out 
#BSUB -e sendmeemail/output_%J.err 

#Load the following in case
#module load python/3.8
module swap cuda/12.0
module swap cudnn/v8.9.1.23-prod-cuda-12.X
module swap python3/3.10.12

python3 train_vae.py \
    --model svhn \
    --svhn_dir ../../Data/SVHN/ \
    --celeba_dir ../../Data/CelebA/ \
    --lr_rate 0.0002 \
    --con_training 0 \
    --train_frac 0.8 \
    --batch_size 100 \
    --epochs 50000 \
    --seed 2712 \
    --save_step 100 \
    --save_path models/
