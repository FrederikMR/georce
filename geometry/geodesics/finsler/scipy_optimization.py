#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.manifolds.finsler import FinslerManifold

#%% Gradient Descent Estimation of Geodesics

class ScipyOptimization(ABC):
    def __init__(self,
                 M:FinslerManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 method:str='BFGS',
                 obj_method:str="finsler",
                 )->None:
        
        if method not in['CG', 'BFGS', 'dogleg', 'trust-ncg', 'trust-exact']:
            raise ValueError(f"Method, {method}, should be gradient based. Choose either: \n CG, BFGS, dogleg, trust-ncg, trust-exact")
            
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
            
        if not (obj_method in ['tensor', 'finsler']):
                raise ValueError(f"The obj_method should be either tensor or finsler. Not {obj_method}.")
        else:
            if obj_method == 'tensor':
                self.obj_fun = M.g
            else:
                self.obj_fun = lambda z,u: M.F(z,u)**2
            
        self.M = M
        self.T = T
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        
        self.save_zt = []
        
        self.dim = None
        self.z0 = None
        self.zT = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Scipy Optimizers"
    
    def energy(self, 
               zt:Array, 
               )->Array:
        
        zt = zt.reshape(-1, self.dim)
        
        term1 = zt[0]-self.z0
        val1 = self.obj_fun(self.z0, term1)
        
        term2 = zt[1:]-zt[:-1]
        val2 = vmap(lambda x,v: self.obj_fun(x,v))(zt[:-1], term2)
        
        term3 = self.zT-zt[-1]
        val3 = self.obj_fun(zt[-1], term3)
        
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
                 step:str="while",
                 )->Array:
        
        self.dim = len(z0)
        zt = self.init_fun(z0,zT,self.T)
        
        self.z0 = z0
        self.zT = zT
        
        #if self.method == "BFGS":
        #    min_fun = jminimize
        #else:
        min_fun = minimize
        
        if step == "while":
            res = min_fun(fun = self.energy, 
                          x0=zt.reshape(-1), 
                          method=self.method, 
                          jac=self.Denergy,
                          tol=self.tol,
                          options={'maxiter': self.max_iter}
                          )
        
            zt = res.x.reshape(-1,self.dim)
            zt = jnp.vstack((z0, zt, zT))
            grad =  res.jac.reshape(-1,self.dim)
            idx = res.nit
        elif step == "for":
            res = min_fun(fun = self.energy,
                          x0=zt.reshape(-1),
                          method=self.method,
                          jac=self.Denergy,
                          callback=self.callback,
                          tol=self.tol,
                          options={'maxiter': self.max_iter}
                          )
            print(len(self.save_zt))
            zt = jnp.stack([zt.reshape(-1,self.dim) for zt in self.save_zt])
            
            grad = vmap(self.Denergy)(zt)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return zt, grad, idx
    
    
    