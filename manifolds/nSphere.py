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

class nSphere(RiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 coordinates="stereographic",
                 )->None:
        
        self.coordinates = coordinates
        if coordinates == "stereographic":
            F = self.F_stereographic
            invF = self.invF_stereographic
        elif coordinates == "spherical":
            F = self.F_spherical
            invF = self.invF_spherical
        else:
            raise ValueError(f"Invalid coordinate system, {coordinates}. Choose either: \n\t-stereographic\n\t-spherical")
        
        self.dim = dim
        self.emb_dim = dim +1
        super().__init__(F=F, invF=invF)
        
        return
    
    def __str__(self)->str:
        
        return f"Sphere of dimension {self.dim} in {self.coordinates} coordinates equipped with the pull back metric"
    
    def F_stereographic(self, 
                     z:Array,
                     )->Array:
        
        s2 = jnp.sum(z**2)
        
        return jnp.hstack(((1-s2), 2*z))/(1+s2)

    def invF_stereographic(self, 
                           x:Array,
                           )->Array:
        
        x0 = x[0]
        return x[1:]/(1+x0)
        
    def F_spherical(self, 
                    z:Array,
                    )->Array:
        
        sin_term = jnp.sin(z)
        x1 = jnp.cos(z[-1])
        prod_term = jnp.cumprod(sin_term[:-1])*x1
        xn = jnp.prod(sin_term)
        
        return jnp.hstack((x1, prod_term, xn))

    def invF_spherical(self, 
                       x:Array,
                       )->Array:
        
        cum_length = jnp.sqrt(jnp.cumsum(x[1::-1]**2))
        
        return vmap(lambda cum, x: jnp.arctan2(cum, x))(cum_length, x[:-1])
    
    def dist(self,
             x:Array,
             y:Array
             )->Array:
        
        return jnp.arccos(jnp.dot(x,y))
    
    def Exp(self,
            x:Array,
            v:Array,
            t:float=1.0,
            )->Array:
        
        norm = jnp.linalg.norm(v)
        
        return jnp.cos(norm*t)*x+jnp.sin(norm*t)*v/norm
    
    def Log(self,
            x:Array,
            y:Array
            )->Array:
        
        dot = jnp.dot(x,y)
        dist = self.dist(x,y)
        val = y-dot*x
        
        return dist*val/jnp.linalg.norm(val)
    
    def Geodesic(self,
                 x:Array,
                 y:Array,
                 t_grid:Array=None,
                 )->Array:
        
        if t_grid is None:
            t_grid = jnp.linspace(0.,1.,100)
        
        v = self.Log(x,y)
        
        return vmap(lambda t: self.Exp(x,v,t))(t_grid)
    
    
    
    