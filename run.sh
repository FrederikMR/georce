    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J FSphere[2, 3, 5, 10, 20, 50, 100]_50
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "rusage[mem=10GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o sendmeemail/error_%J.out 
    #BSUB -e sendmeemail/output_%J.err 
    
    python3 runtime.py \
        --manifold Sphere \
        --geometry Finsler \
        --dim [2, 3, 5, 10, 20, 50, 100] \
        --T 50 \
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
    