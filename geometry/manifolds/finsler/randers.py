#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

####################

from .manifold import FinslerManifold
from geometry.manifolds.riemannian import RiemannianManifold

#%% Code

class RandersManifold(FinslerManifold):
    def __init__(self,
                 RM:RiemannianManifold,
                 b:Callable[[Array], Array],
                 )->None:
        
        self.RM = RM
        self.b = b

        self.dim = dim
        super().__init__(F=self.metric, f=self.RM.f, invF=self.RM.invf)
        
        return
    
    def __str__(self)->str:
        
        return f"Randers manifold of dimension {self.dim} for manifold of type: \n\t-{self.RM.__str__()}"
    
    def metric(self,
               z:Array,
               v:Array,
               )->Array:
        
        g = self.RM.G(z)
        b = self.b(z)
        
        term1 = jnp.einsum('ij,i,j->', g, v, v)
        term2 = jnp.dot(b, v)
        
        return jnp.sqrt(term1)+term2