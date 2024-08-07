#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:42 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp
from jax import jit

import haiku as hk

from geometry.manifolds.riemannian import nSphere, nEllipsoid, nEuclidean, \
    nParaboloid, HyperbolicParaboloid, SPDN, H2, Cylinder, Landmarks, T2

#%% Load manifolds

def load_manifold(manifold:str="Euclidean", 
                  dim:int = 2,
                  ):
    
    rho = 0.5 #default
    if manifold == "Euclidean":
        M = nEuclidean(dim=dim)
        z0 = -jnp.linspace(0,1,dim)
        zT = jnp.ones(dim, dtype=jnp.float32)
        rho = 0.5
    if manifold == "SPDN":
        M = SPDN(N=dim)
        x0 = jnp.eye(dim)
        
        z0 = M.invf(x0)
        zT = jnp.linspace(0.5,1.0, M.dim)
        rho = 0.5
    elif manifold == "Paraboloid":
        M = nParaboloid(dim=dim)
        z0 = -jnp.linspace(0,1,dim)
        zT = jnp.ones(dim, dtype=jnp.float32)
        rho = 0.5
    elif manifold == "Sphere":
        M = nSphere(dim=dim)
        z0 = -jnp.linspace(0,1,dim)
        zT = 0.5*jnp.ones(dim, dtype=jnp.float32)
        rho = .5
    elif manifold == "Ellipsoid":
        params = jnp.linspace(0.5,1.0,dim+1)
        M = nEllipsoid(dim=dim, params=params)
        z0 = -jnp.linspace(0,1,dim)
        zT = 0.5*jnp.ones(dim, dtype=jnp.float32)
        rho = 0.5
    elif manifold == "H2":
        M = H2()
        z0 = jnp.array([1.0,1.0])
        zT = jnp.array([0.1, 0.1])
        rho = 0.5
    elif manifold == "Cylinder":
        M = Cylinder()
        z0 = jnp.array([-5*jnp.pi/4,1.0])
        zT = jnp.array([jnp.pi, -1.0])
        rho = 0.5
    elif manifold == "T2":
        M = T2(R=3.0, r=1.0)
        z0 = jnp.array([0.0, 0.0])
        zT = jnp.array([5*jnp.pi/4, 5*jnp.pi/4])
        rho = 0.5
    elif manifold == "Landmarks":
        M = Landmarks(N=dim, m=2, k_alpha=0.1)
        z0 = jnp.vstack((jnp.linspace(-5.0,5.0,M.N),jnp.linspace(0.0,0.0,M.N))).T.flatten()
        zT = jnp.vstack((jnp.linspace(-1.0,1.0,M.N),jnp.linspace(0.0, 0.0, M.N))).T.flatten()
        rho = 0.5
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")
        
    return z0, zT, M, rho