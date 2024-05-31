#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:52:36 2024

@author: fmry
"""

#%% Sources

#%% Modules

import pickle

#argparse
import argparse

from typing import List

from geodesics import GradientDescent, JAXOptimization, ScipyOptimization, GC_LineSearch
from manifolds import nSphere, nEuclidean, nParaboloid, HyperbolicParaboloid

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

def runtime_geodesics()->None:
    
    args = parse_args()
    output = {}
    for dim in args.dim:
        print(dim)
        x0, xT, M = load_manifold(args.manifold, dim)
        ####################################################
        x0 = jnp.array([-2., -2.])
        xT = jnp.array([2., 2.])
        Geo = GeodesicControl(M=M,
                              tau=args.lr_rate,
                              T=1.0,
                              N=5,
                              tol=args.tol,
                              max_iter=args.max_iter,
                              inv_method=args.inv_method
                              )
        val = Geo(x0,xT,"While")
        print(val)
        return
        ####################################################
        methods = {}
        for method in args.methods:
            sol_method = {}
            if method == "Gradient":
                for optimizer in args.gradient_methods:
                    opt = {}
                    Geo = GradientDescent(M,
                                          tau=args.lr_rate,
                                          T=1.0,
                                          N=args.N,
                                          tol=args.tol,
                                          max_iter=args.max_iter,
                                          optimizer=optimizer
                                          )
                    val = Geo(x0,xT, "While")
                    timing = []
                    for i in range(args.timing_repeats):
                        Geo = GradientDescent(M,
                                              tau=args.lr_rate,
                                              T=1.0,
                                              N=args.N,
                                              tol=args.tol,
                                              max_iter=args.max_iter,
                                              optimizer=optimizer
                                              )
                        start = time.time()
                        _ = Geo(x0,xT, "While")
                        end = time.time()
                        timing.append(end-start)
                    timing = jnp.stack(timing)
                    opt['mu_time'] = jnp.mean(timing)
                    opt['std_time'] = jnp.std(timing)
                    opt['xn'] = val[0]
                    opt['Iter'] = val[1]
                    opt['energy'] = M.energy(val[0])
                    sol_method[optimizer] = opt
            elif method == "ODE":
                opt ={}
                Geo = ODEMinimization(M = M,
                                      bc_tol=args.tol,
                                      N=args.N,
                                      )
                val = Geo(x0,xT)
                timing = []
                for i in range(args.timing_repeats):
                    Geo = ODEMinimization(M = M,
                                          bc_tol=args.tol,
                                          N=args.N,
                                          )
                    start = time.time()
                    _ = Geo(x0,xT)
                    end = time.time()
                    timing.append(end-start)
                timing = jnp.stack(timing)
                opt['mu_time'] = jnp.mean(timing)
                opt['std_time'] = jnp.std(timing)
                opt['xn'] = val[0]
                opt['Iter'] = val[1]
                opt['energy'] = M.energy(val[0])
                sol_method['solve_bvp'] = opt
            elif method == "Scipy":
                for method in args.scipy_methods:
                    opt = {}
                    Geo = ScipyMinimization(M,
                                            T=1.0,
                                            N=args.N,
                                            tol=args.tol,
                                            max_iter=args.max_iter,
                                            method=method,
                                            )
                    val = Geo(x0,xT)
                    timing = []
                    for i in range(args.timing_repeats):
                        Geo = ScipyMinimization(M,
                                                T=1.0,
                                                N=args.N,
                                                tol=args.tol,
                                                max_iter=args.max_iter,
                                                method=method,
                                                )
                        start = time.time()
                        _ = Geo(x0,xT)
                        end = time.time()
                        timing.append(end-start)
                    timing = jnp.stack(timing)
                    opt['mu_time'] = jnp.mean(timing)
                    opt['std_time'] = jnp.std(timing)
                    opt['xn'] = val[0]
                    opt['Iter'] = val[1]
                    opt['energy'] = M.energy(val[0])
                    sol_method[method] = opt
            elif method == "Control":
                opt = {}
                Geo = GeodesicControl(M=M,
                                      tau=args.lr_rate,
                                      T=1.0,
                                      N=args.N,
                                      tol=args.tol,
                                      max_iter=args.max_iter,
                                      inv_method=args.inv_method
                                      )
                val = Geo(x0,xT,"While")
                timing = []
                for i in range(args.timing_repeats):
                    Geo = GeodesicControl(M=M,
                                          tau=args.lr_rate,
                                          T=1.0,
                                          N=args.N,
                                          tol=args.tol,
                                          max_iter=args.max_iter,
                                          inv_method=args.inv_method
                                          )
                    start = time.time()
                    _ = Geo(x0,xT, "While")
                    end = time.time()
                    timing.append(end-start)
                timing = jnp.stack(timing)
                opt['mu_time'] = jnp.mean(timing)
                opt['std_time'] = jnp.std(timing)
                opt['xn'] = val[0]
                opt['Iter'] = val[1]
                opt['energy'] = M.energy(val[0])
                sol_method['control'] = opt
            methods[method] = sol_method
        output[str(dim)] = methods
        
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    save_path = ''.join((args.save_path, args.manifold, '.pkl'))    
    with open(save_path, 'wb') as f:
        pickle.dump(output, f)
    
    return

#%% main

if __name__ == '__main__':
    
    runtime_geodesics()