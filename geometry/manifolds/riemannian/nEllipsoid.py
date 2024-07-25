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

#%% Code

class nEllipsoid(RiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 params:Array=None,
                 coordinates="stereographic",
                 )->None:
        
        if params == None:
            params = jnp.ones(dim+1, dtype=jnp.float32)
        self.params = params
        self.coordinates = coordinates
        if coordinates == "stereographic":
            f = self.f_stereographic
            invf = self.invf_stereographic
        elif coordinates == "spherical":
            f = self.f_spherical
            invf = self.invf_spherical
        else:
            raise ValueError(f"Invalid coordinate system, {coordinates}. Choose either: \n\t-stereographic\n\t-spherical")
        
        self.dim = dim
        self.emb_dim = dim +1
        super().__init__(f=f, invf=invf)
        
        return
    
    def __str__(self)->str:
        
        return f"Sphere of dimension {self.dim} in {self.coordinates} coordinates equipped with the pull back metric"
    
    def f_stereographic(self, 
                     z:Array,
                     )->Array:
        
        s2 = jnp.sum(z**2)
        
        return self.params*jnp.hstack(((1-s2), 2*z))/(1+s2)

    def invf_stereographic(self, 
                           x:Array,
                           )->Array:
        
        x /= self.params
        
        x0 = x[0]
        return x[1:]/(1+x0)
        
    def f_spherical(self, 
                    z:Array,
                    )->Array:
        
        sin_term = jnp.sin(z)
        cos_term = jnp.cos(z)
        
        xn = cos_term[0]
        xi = jnp.cumprod(sin_term[:-1])*cos_term[1:]
        x1 = jnp.prod(sin_term)
        
        return self.params*jnp.hstack((x1, xi, xn))

    def invf_spherical(self, 
                       x:Array,
                       )->Array:
        
        x /= self.params
        
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
        
        x /= self.params
        v /= self.params
        
        norm = jnp.linalg.norm(v)
        
        return (jnp.cos(norm*t)*x+jnp.sin(norm*t)*v/norm)*self.params
    
    def Log(self,
            x:Array,
            y:Array
            )->Array:
        
        x /= self.params
        y /= self.params
        
        dot = jnp.dot(x,y)
        dist = self.dist(x,y)
        val = y-dot*x
        
        return self.params*dist*val/jnp.linalg.norm(val)
    
    
    
    