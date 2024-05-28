#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:42 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp

import jaxman

#%% Load manifolds

def load_manifold(manifold:str="Euclidean", dim = 2):
    
    if manifold == "Euclidean":
        M = jaxman.Euclidean(dim=dim)
        x0 = -0.0*jnp.ones(dim)
        xT = jnp.ones(dim)
    elif manifold == "Paraboloid":
        M = jaxman.Paraboloid(dim=dim)
        x0 = -0.0*jnp.ones(dim)
        xT = jnp.ones(dim)
    elif manifold == "nSphere":
        M = jaxman.nSphere(dim=dim)
        x0 = -0.0*jnp.ones(dim)
        xT = jnp.ones(dim)
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined")
        
    return x0, xT, M