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
from jax import jit, vmap

import timeit

import os

import pickle

#argparse
import argparse

from typing import Dict

#JAX Optimization
from jax.example_libraries import optimizers

from load_manifold import load_manifold
from geometry.manifolds.finsler import RiemannianNavigation
from geometry.geodesics.riemannian import JAXOptimization, ScipyOptimization, GEORCE
from geometry.geodesics.finsler import JAXOptimization as JAXFOptimization
from geometry.geodesics.finsler import ScipyOptimization as ScipyFOptimization
from geometry.geodesics.finsler import GEORCE as GEORCEF

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="celeba",
                        type=str)
    parser.add_argument('--geometry', default="Riemannian",
                        type=str)
    parser.add_argument('--dim', default=32,
                        type=int)
    parser.add_argument('--T', default=100,
                        type=int)
    parser.add_argument('--v0', default=1.5,
                        type=float)
    parser.add_argument('--method', default="GEORCE",
                        type=str)
    parser.add_argument('--jax_lr_rate', default=0.01,
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
    parser.add_argument('--save_path', default='timing/',
                        type=str)
    parser.add_argument('--svhn_path', default="../../../Data/SVHN/",
                        type=str)
    parser.add_argument('--celeba_path', default="../../../Data/CelebA/",
                        type=str)

    args = parser.parse_args()
    return args

#%% Timing

def estimate_method(Geodesic, z0, zT, M, base_length=None):
    
    args = parse_args()
    
    method = {} 
    print("Computing Estimates")
    zt, grad, grad_idx = Geodesic(z0,zT)
    print("\t-Estimate Computed")
    timing = []
    timing = timeit.repeat(lambda: Geodesic(z0,zT)[0].block_until_ready(), 
                           number=args.number_repeats, 
                           repeat=args.timing_repeats)
    print("\t-Timing Computed")
    timing = jnp.stack(timing)
    length = M.length(zt)
    method['grad_norm'] = jnp.linalg.norm(grad)
    method['length'] = length
    method['iterations'] = grad_idx
    method['mu_time'] = jnp.mean(timing)
    method['std_time'] = jnp.std(timing)
    method['tol'] = args.tol
    method['max_iter'] = args.max_iter
    
    if base_length is None:
        method['error'] = None
    else:
        method['error'] = jnp.abs(length-base_length)
    
    return method

#%% Save times

def save_times(methods:Dict, save_path:str)->None:
    
    with open(save_path, 'wb') as f:
        pickle.dump(methods, f)
    
    return

#%% Force Field for Randers manifold

def force_fun(z, M):
    
    val = jnp.cos(z)
    
    val2 = jnp.sqrt(jnp.einsum('i,ij,j->', val, M.G(z), val))
    
    return jnp.sin(z)*val/val2

#%% Riemannian Run Time code

def riemannian_runtime()->None:
    
    args = parse_args()
    
    save_path = ''.join((args.save_path, f'riemannian/{args.manifold}/'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_path = ''.join((save_path, args.method, 
                         f'_{args.manifold}', 
                         f'_d={args.dim}', 
                         f'_T={args.T}.pkl',
                         ))
    if os.path.exists(save_path):
        os.remove(save_path)
    
    z0, zT, M, rho = load_manifold(args.manifold, args.dim,
                                   svhn_path=args.svhn_path,
                                   celeba_path=args.celeba_path,
                                   )
    
    methods = {}
    if hasattr(M, 'Geodesic'):
        #curve = M.Geodesic(z0,zT)
        #true_dist = M.length(curve)
        xt = M.Geodesic(z0,zT)
        zt = vmap(M.invf)(xt)
        length = M.length(zt)
        base_length = length
    else:
        base_length = None
    if args.method == "ground_truth":
        if hasattr(M, 'Geodesic'):
            #curve = M.Geodesic(z0,zT)
            #true_dist = M.length(curve)
            xt = M.Geodesic(z0,zT)
            zt = vmap(M.invf)(xt)
            length = M.length(zt)
            true = {}
            true['length'] = length
            true['grad_norm'] = None
            true['iterations'] = None
            true['mu_time'] = None
            true['std_time'] = None
            true['tol'] = 0.0
            true['max_iter'] = 0
            true['error'] = 0.0
            methods['ground_truth'] = true
            base_length = length
        else:
            true = {}
            true['length'] = None
            true['grad_norm'] = None
            true['iterations'] = None
            true['mu_time'] = None
            true['std_time'] = None
            true['tol'] = None
            true['max_iter'] = None
            true['error'] = None
            methods['ground_truth'] = true
            base_length = None
        save_times(methods, save_path)
    elif args.method == "init":
        Geodesic = GEORCE(M=M,
                          init_fun=None,
                          T=args.T,
                          tol=args.tol,
                          max_iter=args.max_iter,
                          line_search_method="soft",
                          line_search_params={'rho':rho},
                          )
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
        init['error'] = None
        methods['init'] = init
        save_times(methods, save_path)
    elif args.method == "GEORCE":
        Geodesic = GEORCE(M=M,
                          init_fun=None,
                          T=args.T,
                          tol=args.tol,
                          max_iter=args.max_iter,
                          line_search_method="soft",
                          line_search_params={'rho':rho},
                          )
        methods['GEORCE'] = estimate_method(jit(Geodesic), z0, zT, M, base_length)
        save_times(methods, save_path)
    elif args.method == "ADAM":
        Geodesic = JAXOptimization(M = M,
                                   init_fun=None,
                                   lr_rate=args.jax_lr_rate,
                                   optimizer=optimizers.adam,
                                   T=args.T,
                                   max_iter=args.max_iter,
                                   tol=args.tol
                                   )
        methods["ADAM"] = estimate_method(jit(Geodesic), z0, zT, M, base_length)
        save_times(methods, save_path)
    elif args.method == "SGD":
        Geodesic = JAXOptimization(M = M,
                                   init_fun=None,
                                   lr_rate=args.jax_lr_rate,
                                   optimizer=optimizers.sgd,
                                   T=args.T,
                                   max_iter=args.max_iter,
                                   tol=args.tol
                                   )
        methods["SGD"] = estimate_method(jit(Geodesic), z0, zT, M, base_length)
        save_times(methods, save_path)
    else:
        try:
            Geodesic = ScipyOptimization(M = M,
                                         T=args.T,
                                         tol=args.tol,
                                         max_iter=args.max_iter,
                                         method=args.method,
                                         )
            methods[args.method] = estimate_method(Geodesic, z0, zT, M, base_length)
            save_times(methods, save_path)
        except:
            "Method is not defined"
    
    return

#%% Finsler Run Time code

def finsler_runtime()->None:
    
    args = parse_args()
    
    save_path = ''.join((args.save_path, f'finsler/{args.manifold}/'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_path = ''.join((save_path, args.method, 
                         f'_{args.manifold}', 
                         f'_d={args.dim}', 
                         f'_T={args.T}.pkl',
                         ))
    if os.path.exists(save_path):
        os.remove(save_path)
    
    z0, zT, RM, rho = load_manifold(args.manifold, args.dim)
    
    M = RiemannianNavigation(RM=RM,
                             force_fun=lambda z: force_fun(z, RM),
                             v0=args.v0,
                             )
    
    methods = {}
    if hasattr(M, 'Geodesic'):
        #curve = M.Geodesic(z0,zT)
        #true_dist = M.length(curve)
        xt = M.Geodesic(z0,zT)
        zt = vmap(M.invf)(xt)
        length = M.length(zt)
        base_length = length
    else:
        base_length = None
    if args.method == "ground_truth":
        if hasattr(M, 'Geodesic'):
            #curve = M.Geodesic(z0,zT)
            #true_dist = M.length(curve)
            xt = M.Geodesic(z0,zT)
            zt = vmap(M.invf)(xt)
            length = M.length(zt)
            true = {}
            true['length'] = length
            true['grad_norm'] = None
            true['iterations'] = None
            true['mu_time'] = None
            true['std_time'] = None
            true['tol'] = 0.0
            true['max_iter'] = 0
            true['error'] = 0.0
            methods['ground_truth'] = true
            base_length = length
        else:
            true = {}
            true['length'] = None
            true['grad_norm'] = None
            true['iterations'] = None
            true['mu_time'] = None
            true['std_time'] = None
            true['tol'] = None
            true['max_iter'] = None
            true['error'] = None
            methods['ground_truth'] = true
            base_length = None
        save_times(methods, save_path)
    elif args.method == "init":
        Geodesic = GEORCE(M=M,
                          init_fun=None,
                          T=args.T,
                          tol=args.tol,
                          max_iter=args.max_iter,
                          line_search_method="soft",
                          line_search_params={'rho':rho},
                          )
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
        init['error'] = None
        methods['init'] = init
        save_times(methods, save_path)
    elif args.method == "GEORCE":
        Geodesic = GEORCEF(M=M,
                           init_fun=None,
                           T=args.T,
                           tol=args.tol,
                           max_iter=args.max_iter,
                           line_search_method="soft",
                           line_search_params={'rho':rho},
                           )
        methods['GEORCE'] = estimate_method(jit(Geodesic), z0, zT, M, base_length)
        save_times(methods, save_path)
    elif args.method == "ADAM":
        Geodesic = JAXFOptimization(M = M,
                                    init_fun=None,
                                    lr_rate=args.jax_lr_rate,
                                    optimizer=optimizers.adam,
                                    T=args.T,
                                    max_iter=args.max_iter,
                                    tol=args.tol
                                    )
        methods["ADAM"] = estimate_method(jit(Geodesic), z0, zT, M, base_length)
        save_times(methods, save_path)
    elif args.method == "SGD":
        Geodesic = JAXFOptimization(M = M,
                                    init_fun=None,
                                    lr_rate=args.jax_lr_rate,
                                    optimizer=optimizers.sgd,
                                    T=args.T,
                                    max_iter=args.max_iter,
                                    tol=args.tol
                                    )
        methods["SGD"] = estimate_method(jit(Geodesic), z0, zT, M, base_length)
        save_times(methods, save_path)
    else:
        try:
            Geodesic = ScipyFOptimization(M = M,
                                          T=args.T,
                                          tol=args.tol,
                                          max_iter=args.max_iter,
                                          method=args.method,
                                          )
            methods[args.method] = estimate_method(Geodesic, z0, zT, M, base_length)
            save_times(methods, save_path)
        except:
            "Method is not defined"

#%% main

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.geometry == "Riemannian":
        riemannian_runtime()
    elif args.geometry == "Finsler":
        finsler_runtime()
    else:
        raise ValueError("Invalid geometry for runtime comparison.")