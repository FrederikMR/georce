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
        z0 = -jnp.ones(dim, dtype=jnp.float32)
        zT = jnp.ones(dim, dtype=jnp.float32)
    elif manifold == "Paraboloid":
        M = nParaboloid(dim=dim)
        x0 = -0.0*jnp.ones(dim)
        xT = jnp.ones(dim)
    elif manifold == "nSphere":
        M = jaxman.nSphere(dim=dim)
        x0 = -0.0*jnp.ones(dim)
        xT = jnp.ones(dim)
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined")
        
    return x0, xT, M