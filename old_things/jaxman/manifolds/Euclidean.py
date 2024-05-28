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

class Euclidean(RiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 )->None:
        g = lambda x: jnp.eye(dim, dtype=x.dtype) if x.ndim==1 else jnp.tile(jnp.eye(dim, dtype=x.dtype), (x.shape[0], 1,1))
        super().__init__(g=g, F=None)
        self.dim = dim
        
        return
    
    def __str__(self)->str:
        
        return "Euclidean Riemannian manifold with standard metric"