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
from .nEllipsoid import nEllipsoid

#%% Code

class nSphere(nEllipsoid):
    def __init__(self,
                 dim:int=2,
                 coordinates="stereographic",
                 )->None:
        super().__init__(dim=dim, params=jnp.ones(dim+1, dtype=jnp.float32), coordinates=coordinates)
        
        return
    
    def __str__(self)->str:
        
        return f"Sphere of dimension {self.dim} in {self.coordinates} coordinates equipped with the pull back metric"
    
    
    
    