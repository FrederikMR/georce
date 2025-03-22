    #! /bin/bash
    #BSUB -q hpc
    #BSUB -J BVP_LSODA_RSphere1000_100
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "select[model=XeonGold6226R]"
    #BSUB -R "rusage[mem=10GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o sendmeemail/error_%J.out 
    #BSUB -e sendmeemail/output_%J.err 

    module swap python3/3.10.12
    
    python3 runtime.py \
        --manifold Sphere \
        --geometry Riemannian \
        --dim 1000 \
        --T 100 \
        --v0 1.5 \
        --method BVP_LSODA \
        --jax_lr_rate 0.01 \
        --tol 0.0001 \
        --max_iter 1000 \
        --line_search_iter 100 \
        --number_repeats 5 \
        --timing_repeats 5 \
        --seed 2712 \
        --save_path timing_cpu/ \
        --svhn_path /work3/fmry/Data/SVHN/ \
        --celeba_path /work3/fmry/Data/CelebA/
    