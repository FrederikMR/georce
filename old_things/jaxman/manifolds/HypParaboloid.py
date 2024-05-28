#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:10:07 2024

@author: fmry
"""

#%% Modules

from jaxman.setup import *
from jaxman.riemannian import RiemannianManifold

#%% Euclidean Riemannian manifold

class HypParaboloid(RiemannianManifold):
    def __init__(self,
                 )->None:
        F = lambda x: jnp.array([x[0],x[1],x[0]**2-x[1]**2])
        super().__init__(g=None, F=F)
        self.dim = 2
        self.emb_dim = 3
        
        return