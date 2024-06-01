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

class JAXOptimization(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lr_rate:float=1.0,
                 optimizer:Callable=None,
                 T:int=10,
                 max_iter:int=1000,
                 )->None:
        
        self.M = M
        self.T = T
        self.max_iter = max_iter
        
        if optimizer is None:
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr_rate)
        else:
            self.opt_init, self.opt_update, self.get_params = optimizer(lr_rate)
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
            
        self.z0 = None
        self.G0 = None
        self.zT = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def energy(self, 
               zt:Array, 
               )->Array:
        
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
    
    def gradient_step(self, 
                      carry:Tuple[Array, Array],
                      idx:int
                      )->Array:
        
        zt, opt_state = carry
        
        grad = self.Denergy(zt)
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)
        
        return ((zt, opt_state),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 )->Array:
        
        zt = self.init_fun(z0,zT,self.T)
        
        self.z0 = z0
        self.zT = zT
        self.G0 = self.M.G(z0)
        
        grad = self.Denergy(zt)
        opt_state = self.opt_init(zt)
        
        _, val = lax.scan(self.gradient_step,
                          init=(zt, opt_state),
                          xs = jnp.ones(self.max_iter),
                          )
        
        zt = val[0]
        
        return zt