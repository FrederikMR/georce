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
from jax import jit, vmap, lax
import jax.random as jrandom

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
    parser.add_argument('--manifold', default="Paraboloid",
                        type=str)
    parser.add_argument('--n_grid', default=100,
                        type=int)
    parser.add_argument('--runs', default=2,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)
    parser.add_argument('--save_path', default='cut_locus/',
                        type=str)

    args = parser.parse_args()
    return args

#%% Paraboloid

def paraboloid_cut_locus(z0, zT, eps, M):

    @jit
    def compute_dist(e):
        

        Geodesic = jit(GEORCE(M=M,
                              init_fun=lambda x,y,T: e*(zT-z0)+z0,
                              T=100,
                              line_search_method="soft",
                              max_iter=1000,
                              line_search_params={'rho': 0.5},
                             ))
    
        zt = Geodesic(z0,zT)[0]
    
        return zt, jit(M.length)(zt), zt[1]-zt[0]

    zt, dist, u0 = vmap(compute_dist)(eps)
    min_dist = jnp.argmin(dist)
    zt_geodesic = zt[min_dist]

    error = jnp.mean(jnp.linalg.norm(zt-zt_geodesic, axis=0))
    
    cl_dist = lax.cond(error < 1e-1, lambda *_: jnp.min(dist), lambda *_: -1.)

    return cl_dist, zt, u0

#%% Torus

def torus_cut_locus(z0, zT, M):
    
    @jit
    def compute_dist(zT):
    
        zt = Geodesic(z0,zT)[0]
    
        return zt, jit(M.length)(zt), zt[1]-zt[0]

    Geodesic = jit(GEORCE(M=M,
                          init_fun=None,
                          T=100,
                          line_search_method="soft",
                          max_iter=1000,
                          line_search_params={'rho': 0.5},
                          ))
    zT = jnp.vstack((zT,
                     jnp.array([zT[0]-2.*jnp.pi, zT[1]]),
                     jnp.array([zT[0], zT[1]-2.*jnp.pi]),
                     zT-2.*jnp.pi,
                    ))

    zt, dist, u0 = vmap(compute_dist)(zT)
    dist_array = jnp.abs(dist-jnp.min(dist))
    
    cl_dist = lax.cond(jnp.sum(dist_array < 1e-1) < 2, lambda *_: jnp.min(dist), lambda *_: -1.)
    
    return cl_dist, zt, u0

#%% Grid Computers

def paraboloid_grid_fun(n_points:int=100):
        
    x1 = jnp.linspace(-2.0, 2.0, n_points)
    x2 = jnp.linspace(-2.0,2.0, n_points)
    
    X1, X2 = jnp.meshgrid(x1,x2)
    
    return X1, X2

def torus_grid_fun(n_points:int=100):
        
    U = jnp.linspace(0, 2*jnp.pi, n_points)
    V = jnp.linspace(0, 2*jnp.pi, n_points)
    U, V = jnp.meshgrid(U, V)
    
    return U, V

#%% Save times

def save_times(methods:Dict, save_path:str)->None:
    
    with open(save_path, 'wb') as f:
        pickle.dump(methods, f)
    
    return

#%% Riemannian Run Time code

def compute_cut_locus()->None:
    
    args = parse_args()
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_path = ''.join((save_path,
                         f'{args.manifold}.pkl',
                         ))

    if args.manifold == "Paraboloid":
        z0 = jnp.ones(2, dtype=jnp.float32)
        _, _, M, _ = load_manifold("Paraboloid", dim=2)
        
        key = jrandom.key(args.seed)
        key, subkey = jrandom.split(key)
        eps = jrandom.normal(subkey, shape=(args.runs,100-1,2))
        
        z1, z2 = paraboloid_grid_fun(args.n_grid)
        
        cl_lst = []
        zt_lst = []
        u0_lst = []
        for i in range(1, 2+args.n_grid//100):
            val = vmap(vmap(lambda u,v: paraboloid_cut_locus(z0,jnp.stack((u,v)),eps,M)))(z1[((i-1)*100):(i*100)],
                                                                                          z2[((i-1)*100):(i*100)])
            cl_lst.append(val[0])
            zt_lst.append(val[1])
            u0_lst.append(val[2])
            
        cl = jnp.concatenate(cl_lst, axis=0)
        zt = jnp.concatenate(zt_lst, axis=0)
        u0 = jnp.concatenate(u0_lst, axis=0)
        
        save_times({'cl': cl, 'zt': zt, 'u0': u0}, save_path) 
    elif args.manifold == "T2":
        
        z0 = jnp.zeros(2, dtype=jnp.float32)
        _, _, M, _ = load_manifold("T2", dim=2)
        
        z1, z2 = torus_grid_fun(args.n_grid)
        
        cl_lst = []
        zt_lst = []
        u0_lst = []
        for i in range(1, 2+args.n_grid//100):
            val = vmap(vmap(lambda u,v: torus_cut_locus(z0,jnp.stack((u,v)),M)))(z1[((i-1)*100):(i*100)],
                                                                                 z2[((i-1)*100):(i*100)])
            cl_lst.append(val[0])
            zt_lst.append(val[1])
            u0_lst.append(val[2])
        
        cl = jnp.concatenate(cl_lst, axis=0)
        zt = jnp.concatenate(zt_lst, axis=0)
        u0 = jnp.concatenate(u0_lst, axis=0)
        
        save_times({'cl': cl, 'zt': zt, 'u0': u0}, save_path)
    else:
        raise ValueError(f"{args.manifold} is invalid manifold.")

    return

#%% main

if __name__ == '__main__':
    
    args = parse_args()
    
    compute_cut_locus()
    