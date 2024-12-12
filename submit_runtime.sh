    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J init_Fmnist8_100
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=10GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o sendmeemail/error_%J.out 
    #BSUB -e sendmeemail/output_%J.err 
    
    module swap cuda/12.0
    module swap cudnn/v8.9.1.23-prod-cuda-12.X
    module swap python3/3.10.12
    
    python3 runtime.py \
        --manifold mnist \
        --geometry Finsler \
        --dim 8 \
        --T 100 \
        --v0 1.5 \
        --method init \
        --jax_lr_rate 0.01 \
        --tol 0.001 \
        --max_iter 1000 \
        --line_search_iter 100 \
        --number_repeats 5 \
        --timing_repeats 5 \
        --seed 2712 \
        --save_path timing_gpu/ \
        --svhn_path /work3/fmry/Data/SVHN/ \
        --celeba_path /work3/fmry/Data/CelebA/
    