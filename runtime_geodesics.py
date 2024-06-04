#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:52:36 2024

@author: fmry
"""

#%% Sources

#https://jax.readthedocs.io/en/latest/faq.html

#%% Modules

import jax.numpy as jnp
from jax import jit

import timeit

import os

import pickle

#argparse
import argparse

from typing import List, Dict

#JAX Optimization
from jax.example_libraries import optimizers

from load_manifold import load_manifold
from geodesics import GradientDescent, JAXOptimization, ScipyOptimization, GC_LineSearch

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="Sphere",
                        type=str)
    parser.add_argument('--dim', default=2,
                        type=int)
    parser.add_argument('--T', default=100,
                        type=int)
    parser.add_argument('--scipy_methods', default=1,
                        type=int)
    parser.add_argument('--jax_methods', default=1,
                        type=int)
    parser.add_argument('--jax_lr_rate', default=0.01,
                        type=float)
    parser.add_argument('--gc_lr_rate', default=1.0,
                        type=float)
    parser.add_argument('--gradient_lr_rate', default=1.0,
                        type=float)
    parser.add_argument('--gc_decay_rate', default=0.5,
                        type=float)
    parser.add_argument('--gradient_decay_rate', default=0.5,
                        type=float)
    parser.add_argument('--tol', default=1e-4,
                        type=float)
    parser.add_argument('--max_iter', default=1000,
                        type=int)
    parser.add_argument('--line_search_iter', default=100,
                        type=int)
    parser.add_argument('--number_repeats', default=5,
                        type=int)
    parser.add_argument('--timing_repeats', default=5,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)
    parser.add_argument('--save_path', default='../timing/',
                        type=str)

    args = parser.parse_args()
    return args

#%% Timing

def estimate_method(Geodesic, z0, zT, M):
    
    args = parse_args()
    
    method = {} 
    zt, grad, grad_idx = Geodesic(z0,zT)
    timing = []
    timing = timeit.repeat(lambda: Geodesic(z0,zT)[0].block_until_ready(), 
                           number=args.number_repeats, 
                           repeat=args.timing_repeats)
    timing = jnp.stack(timing)
    length = M.length(zt)
    method['grad_norm'] = jnp.linalg.norm(grad)
    method['length'] = length
    method['iterations'] = grad_idx
    method['mu_time'] = jnp.mean(timing)
    method['std_time'] = jnp.std(timing)
    method['tol'] = args.tol
    method['max_iter'] = args.max_iter
    
    return method

#%% Save times

def save_times(methods:Dict, save_path:str)->None:
    
    with open(save_path, 'wb') as f:
        pickle.dump(methods, f)
    
    return


#%% Run Time code

def runtime_geodesics()->None:
    
    args = parse_args()
    
    jax_methods = {"adam": optimizers.adam, "sgd": optimizers.sgd}
    scipy_methods = ["BFGS"]
    
    save_path = ''.join((args.save_path, args.manifold, '/'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_path = ''.join((save_path, args.manifold, str(args.dim), '.pkl'))
    
    z0, zT, M = load_manifold(args.manifold, args.dim)
    methods = {}
    ## Gradient descent
    print("Computing Gradient Descent")
    Geodesic = GradientDescent(M = M,
                               init_fun=None,
                               lr_rate=args.gradient_lr_rate,
                               decay_rate=args.gradient_decay_rate,
                               T=args.T,
                               max_iter=args.max_iter,
                               tol=args.tol,
                               line_search_iter=args.line_search_iter
                               )
    methods['gradient'] = estimate_method(jit(Geodesic), z0, zT, M)
    ## Init Length
    zt = Geodesic.init_fun(z0,zT,args.T)
    init_length = M.length(zt)
    init = {}
    init['length'] = init_length
    init['grad_norm'] = None
    init['iterations'] = None
    init['mu_time'] = None
    init['std_time'] = None
    init['tol'] = args.tol
    init['max_iter'] = args.max_iter
    methods['init'] = init
    save_times(methods, save_path)
    ## True Length
    if hasattr(M, 'dist'):
        true_dist = M.dist(z0,zT)
        true = {}
        true['length'] = true_dist
        true['grad_norm'] = None
        true['iterations'] = None
        true['mu_time'] = None
        true['std_time'] = None
        true['tol'] = args.tol
        true['max_iter'] = args.max_iter
        methods['ground_truth'] = true
    else:
        true = {}
        true['length'] = None
        true['grad_norm'] = None
        true['iterations'] = None
        true['mu_time'] = None
        true['std_time'] = None
        true['tol'] = args.tol
        true['max_iter'] = args.max_iter
        methods['ground_truth'] = true
    save_times(methods, save_path)
    ## Geodesic Control
    print("Computing Geodesic Control")
    Geodesic = GC_LineSearch(M=M,
                             init_fun=None,
                             lr_rate=args.gc_lr_rate,
                             T=args.T,
                             decay_rate=args.gc_decay_rate,
                             tol=args.tol,
                             max_iter=args.max_iter,
                             line_search_iter=args.line_search_iter
                             )
    methods['geodesic_control'] = estimate_method(jit(Geodesic), z0, zT, M)
    save_times(methods, save_path)
    ## JAX
    if args.jax_methods:
        print("Computing JAX")
        for opt in jax_methods:
            Geodesic = JAXOptimization(M = M,
                                       init_fun=None,
                                       lr_rate=args.jax_lr_rate,
                                       optimizer=jax_methods[opt],
                                       T=args.T,
                                       max_iter=args.max_iter,
                                       tol=args.tol
                                       )
            methods[opt] = estimate_method(jit(Geodesic), z0, zT, M)
            save_times(methods, save_path)
    ## Scipy
    if args.scipy_methods:
        print("Computing Scipy")
        for m in scipy_methods:
            Geodesic = ScipyOptimization(M = M,
                                         T=args.T,
                                         tol=args.tol,
                                         max_iter=args.max_iter,
                                         method=m,
                                         )
            methods[m] = estimate_method(Geodesic, z0, zT, M)
            save_times(methods, save_path)
    
    return

#%% main

if __name__ == '__main__':
    
    runtime_geodesics()