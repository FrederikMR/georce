#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:58:42 2024

@author: fmry
"""

#%% Modules

from jaxman.setup import *
from jaxman.riemannian import RiemannianManifold

from scipy.optimize import minimize

#%% Scipy Minmization

class ScipyMinimization(object):
    def __init__(self,
                 M:RiemannianManifold,
                 T:float=1.0,
                 N:int=100,
                 tol:float=1e-3,
                 max_iter:int=1000,
                 method:str='BFGS',
                 )->None:
        
        if method not in['CG', 'BFGS', 'dogleg', 'trust-ncg', 'trust-exact']:
            raise ValueError(f"Method, {method}, should be gradient based. Choose either: \n CG, BFGS, dogleg, trust-ncg, trust-exact")
            
        self.M = M
        self.T = T
        self.N = N
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Scipy Minimization"
    
    def __call__(self, x0:Array, xT:Array)->Array:
        
        t_grid = jnp.linspace(0,self.T,self.N)
        xn = x0+t_grid.reshape(-1,1)[1:-1]*(xT-x0)
        dim = len(x0)
        
        obj_fun = lambda x: self.M.energy(jnp.concatenate((x0.reshape(1,-1),xn.reshape(-1,dim),xT.reshape(1,-1)), axis=0))
        
        res = minimize(obj_fun, xn.reshape(-1), options={'gtol': self.tol, 'maxiter': self.max_iter})
        
        return jnp.concatenate((x0.reshape(1,-1),res.x.reshape(-1,dim),xT.reshape(1,-1))), res.nit