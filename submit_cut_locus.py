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
    
    os.system("bsub < submit_cl.sh")
    
    return

#%% Generate jobs

def generate_job(manifold):

    with open ('submit_cl.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J {manifold}
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
    
    python3 runtime.py \\
        --manifold {manifold} \\
        --n_grid 100 \\
        --runs 10 \\
        --śeed 2712 \\
        --save_path cut_locus/ \\
    ''')
    
    return

#%% Loop jobs

def loop_jobs(wait_time = 1.0):
    
    manifolds = ['Paraboloid', 'T2']
    
    for man in manifolds:
        time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
        generate_job(man)
        try:
            submit_job()
        except:
            time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
            try:
                submit_job()
            except:
                print(f"Job script with {man} failed!")

#%% main

if __name__ == '__main__':
    
    loop_jobs(1.0)