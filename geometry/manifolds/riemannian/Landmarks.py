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

class Landmarks(RiemannianManifold):
    def __init__(self,
                 N:int=1, #number of landmarks
                 m:int=2, #dimension of landmarks
                 k_alpha:float=1.0,
                 k_sigma:Array=None,
                 k_fun:Callable[[Array], Array]=None,
                 )->None:
        
        self.N = N
        self.m = m
        
        self.k_alpha = k_alpha
        
        if k_sigma is None:
            self.k_sigma = jnp.eye(self.m)
        else:
            self.k_sigma = k_sigma
        self.inv_k_sigma = jnp.linalg.inv(self.k_sigma)
        self.k_Sigma = jnp.tensordot(self.k_sigma,self.k_sigma,(1,1))
        
        if k_fun is None:
            self.k_fun = lambda x: self.k_alpha*jnp.exp(-.5*jnp.square(jnp.tensordot(x,self.inv_k_sigma,(x.ndim-1,1))).sum(x.ndim-1))
        else:
            self.k_fun = k_fun
        
        self.dim = self.m*self.N
        self.rank = self.dim
        
        # kernel differentials
        self.dk = grad(self.k_fun)
        self.d2k = hessian(self.k_fun)

        # in coordinates
        self.k_q = lambda q1,q2: self.k_fun(q1.reshape((-1,self.m))[:,jnp.newaxis,:]-q2.reshape((-1,self.m))[jnp.newaxis,:,:])
        self.K = lambda q1,q2: (self.k_q(q1,q2)[:,:,jnp.newaxis,jnp.newaxis]*jnp.eye(self.m)[jnp.newaxis,jnp.newaxis,:,:]).transpose((0,2,1,3)).reshape((q1.size,q2.size))

        super().__init__(G=self.g, f=self.f_standard, invf=self.invf_standard)
        
        return
    
    def __str__(self)->str:
        
        return "Landmarks space with LDDMM metric"
    
    def gsharp(self, z:Array):
        
        z = z.reshape(self.N, self.m)
        
        return self.K(z,z) 
    
    def g(self, z:Array):
        
        return jnp.linalg.inv(self.gsharp(z))
    
    def f_standard(self,
                   z:Array,
                   )->Array:
        
        return z.reshape(self.N, self.m)
    
    def invf_standard(self,
                      x:Array,
                      )->Array:
        
        return x.reshape(self.N, self.m)
        