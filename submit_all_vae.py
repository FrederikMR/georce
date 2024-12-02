#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:22:08 2024

@author: fmry
"""

#%% Modules

import numpy as np

import os

import time

#%% Submit job

def submit_job():
    
    os.system("bsub < submit_vae.sh")
    
    return

#%% Generate jobs

def generate_job(model:str, latent_dim:int,):

    with open ('submit_vae.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J {model}_{latent_dim}
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
    
    python3 train_vae.py \\
        --model {model} \\
        --svhn_dir /work3/fmry/Data/SVHN/ \\
        --celeba_dir /work3/fmry/Data/CelebA/ \\
        --lr_rate 0.0002 \\
        --con_training 0 \\
        --split 0.8 \\
        --batch_size 100 \\
        --latent_dim {latent_dim} \\
        --epochs 500000 \\
        --save_step 10000 \\
        --save_path models/ \\
        --seed 2712
    ''')
    
    return

#%% Loop jobs

def loop_jobs(wait_time = 1.0):
    
    models = {'mnist': [8], 
              'svhn': [32], 
              'celeba': [32],
              }
    
    for model,dims in models.items():
        for ldim in dims:
            time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
            generate_job(model, ldim)
            try:
                submit_job()
            except:
                time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                try:
                    submit_job()
                except:
                    print(f"Job script for {model} with latent dim {ldim} failed!")
                    
    return

#%% main

if __name__ == '__main__':
    
    loop_jobs(1.0)