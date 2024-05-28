#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

from setup import *

####################

from .riemannian import RiemannianManifold

#%% Code

class nParaboloid(RiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 )->None:

        self.dim = dim
        self.emb_dim = dim+1
        super().__init__(F=self.F_standard, invF=self.invF_standard)
        
        return
    
    def __str__(self)->str:
        
        return f"Paraboloid of dimension {self.dim} equipped with the pull back metric"
    
    def F_standard(self,
                   z:Array,
                   )->Array:
        
        return jnp.hstack((z, jnp.sum(z**2)))

    def invF_standard(self,
                      x:Array,
                      )->Array:
        
        return x[:-1]
        