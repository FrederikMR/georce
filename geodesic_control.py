#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:49:36 2024

@author: frederik
"""

#%% Sources

#%% Modules

import jax.numpy as jnp
from jax import Array, vmap, lax, jacfwd
from jax.scipy.optimize import minimize

from typing import Callable, Tuple

#%% Inverse Matrix Computations

class InvMat(object):
    def __init__(self, 
                 matrix_fun:Callable[[Array], Array],
                 method:str = 'Naive',
                 )->None:
        
        self.matrix_fun = matrix_fun
        self.method = method
        
        return
    
    def __str__(self)->str:
        
        return "Object Computing Inverse Matrix"
    
    def __call__(self, 
                 x:Array
                 )->Array:
        
        if self.method == "Naive":
            return jnp.linalg.inv(self.matrix_fun(x))
        else:
            raise ValueError(f"Method {self.method} is not implemented. Choose: Naive")
            
#%% Line Search

class LineSearch(object):
    def __init__(self,
                 loss_fun:Callable[[Array], Array],
                 alpha:float=1.0
                 )->None:
        
        self.loss_fun = loss_fun
        self.alpha = alpha
        
        return
    
    def __str__(self)->str:
        
        return "Object Computing Line Search"
            
#%% Geodesic Control

class GeodesicOptimization(object):
    def __init__(self,
                 g:Callable[[Array], Array],
                 Dg:Callable[[Array], Array] = None,
                 solver:Callable[[Callable, Array, Tuple], Array] = lambda fun, x0, args: minimize(fun, x0, args=args, method='BFGS',
                                                                                                   tol=1e-3,
                                                                                                   options={'maxiter': 100}),
                 alpha:float=1.0,
                 inv_method:str = 'Naive',
                 max_iter:int=1000,
                 tol:float=1e-5,
                 )->None:
        
        self.g = g
        if Dg is None:
            self.Dg = jacfwd(g)
        else:
            self.Dg = Dg
        self.ginv = InvMat(g, inv_method)
        self.solver=solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Optimization Object"
    
    def energy(self, 
               xt:Array, 
               )->Array:
        
        
        G = vmap(lambda x: self.g(x))(xt[:-1])
        xdiff = xt[1:]-xt[:-1]
        
        return jnp.sum(jnp.einsum('...i,...ij,...j->...', xdiff, G, xdiff), axis=0)
    
    def algorithm_a_iter(self,
                         x0:Array,
                         xT:Array,
                         n_points:int=100
                         )->Array:
        
        def obj_fun(mut, *args)->Array:
            
            g_cumsum = args[0]
            ginv = args[1]
            
            loss = jnp.sum(jnp.einsum('...ij,...j->...i', ginv, mut+g_cumsum), axis=0)+diff
            
            return jnp.dot(loss, loss)
        
        def step(carry, idx):
            
            xt, ut, mut = carry
            
            ginv = vmap(lambda x: self.ginv(x))(xt[:-1])
            Dg = vmap(lambda x: self.Dg(x))(xt[1:-1])
            Dginv = jnp.einsum('...jk,...kld,...li->...jid', ginv[1:], Dg, ginv[1:])
            gt = jnp.einsum('...j,...jid,...i->...d', ut[1:], Dginv, ut[1:])
            g_cumsum = jnp.cumsum(gt)
            
            g_invsum = self.ginv(jnp.sum(ginv, axis=0))
            mut = jnp.dot(g_invsum, 0.5*jnp.sum(jnp.einsum('...ij,...j->...i', ginv, g_cumsum), axis=0)+xT-x0)
            mut = mut+g_cumsum[::-1]
            
            uhat = -0.5*jnp.einsum('...ij,...j->...i', ginv, mut)
            
            ut = self.alpha*uhat+(1-self.alpha)*ut
            xt = jnp.cumsum(xt+self.alpha*uhat+(1-self.alpha)*ut)
            
            return ((xt, ut, mut),)*2
        
        T = n_points-1
        dim = len(x0)
        diff = xT-x0
        
        t_grid = jnp.round(jnp.linspace(0,1,T+1)*T) #jnp.arrange(0,T,1)
        diff_scaled = (xT-x0)/T
        xt = jnp.einsum('i,j->ji', x0+diff_scaled, t_grid)
        ut = jnp.tile(diff_scaled, (T,1))
        
        lax.scan(step, 
                 init=(xt,ut,jnp.zeros(dim, dtype=x0.dtype)),
                 xs=jnp.ones(self.max_iter)
                 )
        
    def algorithm_a_tol(self,
                        x0:Array,
                        xT:Array,
                        n_points:int=100
                        )->Array:
        
        def obj_fun(mut, g_cumsum, ginv)->Array:
            
            loss = jnp.sum(jnp.einsum('...ij,...j->...i', ginv, mut+g_cumsum), axis=0)+diff
            
            return jnp.dot(loss, loss)
        
        def step(carry, idx):
            
            xt, ut, mut = carry
            
            ginv = vmap(lambda x: self.ginv(x))(xt[1:])
            Dg = vmap(lambda x: self.Dg(x))(xt[1:])
            Dginv = jnp.einsum('...jk,...kld,...li->...jid', ginv, Dg)
            gt = jnp.einsum('...j,...jid,...i->...d', ut, Dginv, ut)
            g_cumsum = jnp.cumsum(gt)
            
            
            mut = self.solver(obj_fun, mut, (g_cumsum, ginv))
            mut += g_cumsum
            
            uhat = -0.5*jnp.einsum('...ij,...j->...i', ginv, mut)
            alpha = 1.0
            
            ut = alpha*uhat+(1-alpha)*ut
            xt = jnp.cumsum(xt+alpha*uhat+(1-alpha)*ut)
            
            return ((xt, ut, mut),)*2
        
        T = n_points-1
        dim = len(x0)
        diff = xT-x0
        
        t_grid = jnp.round(jnp.linspace(0,1,T)*T) #jnp.arrange(0,T,1)
        diff_scaled = (xT-x0)/T
        xt = jnp.einsum('i,j->ji', x0+diff_scaled, t_grid)
        ut = jnp.tile(diff_scaled, (T,1))
        
        lax.scan(step, init=(xt,ut,jnp.zeros(dim, dtype=x0.dtype)))
        
        
        
        return