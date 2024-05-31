#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:42 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp

from manifolds import nSphere, nEuclidean, nParaboloid, HyperbolicParaboloid

#%% Load manifolds

def load_manifold(manifold:str="Euclidean", dim = 2):
    
    if manifold == "Euclidean":
        M = nEuclidean(dim=dim)
        z0 = -jnp.linspace(0,1,2)
        zT = jnp.ones(2, dtype=jnp.float32)
    elif manifold == "Paraboloid":
        M = nParaboloid(dim=dim)
        z0 = -jnp.linspace(0,1,2)
        zT = jnp.ones(2, dtype=jnp.float32)
    elif manifold == "Sphere":
        M = nSphere(dim=dim)
        z0 = -jnp.linspace(0,1,dim)
        zT = 0.5*jnp.ones(dim, dtype=jnp.float32)
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")
        
    return z0, zT, M