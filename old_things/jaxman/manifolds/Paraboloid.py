#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:10:07 2024

@author: fmry
"""

#%% Modules

from jaxman.setup import *
from jaxman.riemannian import RiemannianManifold
from jaxman.optimization import GeodesicControl, ScipyMinimization, ODEMinimization, GradientDescent

#%% Euclidean Riemannian manifold

class Paraboloid(RiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 )->None:
        F = lambda x: jnp.hstack((x,jnp.sum(x**2,axis=-1).reshape(1))) if x.ndim == 1 else \
            jnp.hstack((x,jnp.sum(x**2,axis=-1).reshape(-1,1)))
        super().__init__(g=None, F=F)
        self.dim = dim
        self.emb_dim = dim+1
        
        return
    
    def __str__(self)->str:
        
        return "Paraboloid Riemannian manifold with standard metric"