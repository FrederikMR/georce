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

class nSphere(RiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 )->None:
        def F_steographic(x):
            
            s2 = jnp.sum(x**2)
            val = jnp.concatenate(((1-s2).reshape(1), 2*x))/(1+s2)
                
            return val
    
        def invF_steographic(x):
            
            x0 = x[0]
            
            val = vmap(lambda xi: xi/(1+x0))(x[1:])
            
            return val
        super().__init__(g=None, F=F_steographic)
        self.dim = dim
        self.emb_dim = dim+1
        
        return
    
    def __str__(self)->str:
        
        return "Paraboloid Riemannian manifold with standard metric"