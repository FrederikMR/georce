#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from setup import *

from manifolds import RiemannianManifold

#%% Gradient Descent Estimation of Geodesics

class ScipyOptimization(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-8,
                 max_iter:int=1000,
                 method:str='BFGS',
                 )->None:
        
        if method not in['CG', 'BFGS', 'dogleg', 'trust-ncg', 'trust-exact']:
            raise ValueError(f"Method, {method}, should be gradient based. Choose either: \n CG, BFGS, dogleg, trust-ncg, trust-exact")
            
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
            
        self.M = M
        self.T = T
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        
        self.save_zt = []
        
        self.dim = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Scipy Optimizers"
    
    def energy(self, 
               zt:Array, 
               )->Array:
        
        zt = zt.reshape(-1,self.dim)
        
        term1 = zt[0]-self.z0
        val1 = jnp.einsum('i,ij,j->', term1, self.G0, term1)
        
        term2 = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.M.G(z))(zt)
        val2 = jnp.einsum('ti,tij,tj->t', term2, Gt[:-1], term2)
        
        term3 = self.zT-zt[-1]
        val3 = jnp.einsum('i,ij,j->', term3, Gt[-1], term3)
        
        return val1+jnp.sum(val2)+val3
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        return grad(lambda z: self.energy(z))(zt)
    
    def callback(self,
                 zt:Array
                 )->Array:
        
        self.save_zt.append(zt.reshape(-1, self.dim))
        
        return
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 )->Array:
        
        self.dim = len(z0)
        zt = self.init_fun(z0,zT,self.T)
        
        self.z0 = z0
        self.zT = zT
        self.G0 = self.M.G(z0)
        
        res = minimize(fun = self.energy, 
                       x0=zt.reshape(-1), 
                       method=self.method, 
                       jac=self.Denergy,
                       callback=self.callback)
        
        zt = self.save_zt
        self.save_zt = None
        
        return jnp.stack(zt)
    