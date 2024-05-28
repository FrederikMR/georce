#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:58:42 2024

@author: fmry
"""

#%% Modules

from jaxman.setup import *
from jaxman.riemannian import RiemannianManifold

from scipy.integrate import solve_bvp

#%% Solve BVP Problem

class ODEMinimization(object):
    def __init__(self,
                 M:RiemannianManifold=None,
                 bc_tol:float=1e-3,
                 T:float=1.0,
                 N:int=100,
                 )->None:
        
        self.M = M
        self.bc_tol = bc_tol
        self.T = T
        self.N = N
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using ODE Solver"
    
    def __call__(self, x0:Array, xT:Array)->Array:
        
        dim = len(x0)
        obj_fun = lambda x,y: self.M.euler_lagrange(y[dim:], y[0:dim])
        bc_fun = lambda ya,yb: jnp.hstack((ya[:dim]-x0, yb[:dim]-xT))
        
        t = jnp.linspace(0,self.T,self.N)
        y = jnp.zeros((2*dim, self.N))
        
        res = solve_bvp(obj_fun, bc_fun, t, y, bc_tol=self.bc_tol)
        
        return res.sol(t)[:dim].T, res.niter