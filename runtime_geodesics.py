#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:52:36 2024

@author: fmry
"""

#%% Sources

#%% Modules

#argparse
import argparse

from typing import List

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="Sphere",
                        type=str)
    parser.add_argument('--dim', default=[2,3,5,10,20],
                        type=List)
    parser.add_argument('--T', default=100,
                        type=int)
    parser.add_argument('--scipy_methods', default=["BFGS"],
                        type=List)
    parser.add_argument('--jax_methods', default=["Adam"],
                        type=List)
    parser.add_argument('--inv_method', default="Naive",
                        type=str)
    parser.add_argument('--jax_lr_rate', default=[0.01],
                        type=List)
    parser.add_argument('--gc_lr_rate', default=1.0,
                        type=float)
    parser.add_argument('--gradient_lr_rate', default=1.0,
                        type=float)
    parser.add_argument('--gc_decay_rate', default=0.99,
                        type=float)
    parser.add_argument('--gradient_decay_rate', default=0.95,
                        type=float)
    parser.add_argument('--tol', default=1e-4,
                        type=float)
    parser.add_argument('--max_iter', default=1000,
                        type=int)
    parser.add_argument('--timing_repeats', default=5,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)
    parser.add_argument('--save_path', default='../timing/',
                        type=str)

    args = parser.parse_args()
    return args

#%% Run Time code

def runtime_geodesics():
    
    return

#%% main

if __name__ == '__main__':
    
    runtime_geodesics()