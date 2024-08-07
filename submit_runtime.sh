#!/bin/sh
#BSUB -q gpuv100
#BSUB -J RT2_100
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

python3 runtime.py \
    --manifold T2 \
    --geometry Riemannian \
    --dim 2 \
    --T 100 \
    --v0 1.5 \
    --scipy_methods 1 \
    --jax_methods 1 \
    --jax_lr_rate 0.01 \
    --tol 1e-4 \
    --max_iter 1000 \
    --line_search_iter 100 \
    --number_repeats 5 \
    --timing_repeats 5 \
    --seed 2712 \
    --save_path timing/
