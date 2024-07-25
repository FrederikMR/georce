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

from .manifold import RiemannianManifold
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
    
    def Geodesic(self,
                 x:Array,
                 y:Array,
                 t_grid:Array=None,
                 )->Array:
        
        if t_grid is None:
            t_grid = jnp.linspace(0.,1.,99, endpoint=False)[1:]
        
        x = self.f(x)
        y = self.f(y)
        
        x_s = x/self.params
        y_s = y/self.params
        
        v = self.Log(x,y)
        
        gamma = self.params*vmap(lambda t: self.Exp(x_s, v,t))(t_grid)
        
        return jnp.vstack((x,gamma,y))
    
    
    