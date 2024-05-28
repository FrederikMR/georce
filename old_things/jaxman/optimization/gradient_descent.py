#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:28:32 2024

@author: fmry
"""

#%% Sources

#%% Modules

from jaxman.setup import *
from jaxman.riemannian import RiemannianManifold

#JAX Optimization
from jax.example_libraries import optimizers

#%% Gradient Descent

class GradientDescent(object):
    def __init__(self,
                 M:RiemannianManifold,
                 tau:float=1.0,
                 T:float=1.0,
                 N:int=100,
                 tol:float=1e-3,
                 max_iter:int=1000,
                 optimizer:str="Adam",
                 )->None:
        
        self.M = M
        self.tau = tau
        self.T = T
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        
        if optimizer == "Adam":
            opt_init, opt_update, get_params = optimizers.adam(tau)
        elif optimizer == "SGD":
            opt_init, opt_update, get_params = optimizers.sgd(tau)
            
        self.opt_init = opt_init
        self.opt_update = opt_update
        self.get_params = get_params
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Gradient Descent"
    
    def __call__(self, x0:Array, xT:Array, method="While")->Array:
        
        def gradient_for(carry, idx):
            
            xn, opt_state = carry
            grad = grad_fun(xn)
            
            opt_state = self.opt_update(idx, grad, opt_state)
            xn = self.get_params(opt_state)
            
            return ((xn, opt_state), xn)
        
        def gradient_while(val):
            
            xn, grad, opt_state, idx = val
            
            opt_state = self.opt_update(idx, grad, opt_state)
            xn = self.get_params(opt_state)
            grad = grad_fun(xn)
            
            return (xn, grad, opt_state, idx+1)

        self.diff = xT-x0
        t_grid = jnp.linspace(0,self.T,self.N)
        xn = x0+t_grid.reshape(-1,1)[1:-1]*self.diff
        
        #grad_fun = lambda x: jacfwd(lambda y: self.M.energy(jnp.concatenate((x0.reshape(1,-1),
        #                                                                     y,
        #                                                                     xT.reshape(1,-1)), axis=0)))(x)
        grad_fun = lambda x: self.M.Denergy(jnp.concatenate((x0.reshape(1,-1),
                                                             x,
                                                             xT.reshape(1,-1)), axis=0))
        opt_state = self.opt_init(xn)
        
        if method == "For":
            val, carry = lax.scan(gradient_for, init = (xn, opt_state), xs = jnp.arange(0,self.max_iter,1))
            return val, carry
        else:
            grad = grad_fun(xn)
            cond_fun = lambda val: ((jnp.linalg.norm(val[1])>self.tol)*(val[-1]<self.max_iter))
            val = lax.while_loop(cond_fun, gradient_while, init_val=(xn, grad, opt_state, 0))
            xn = jnp.concatenate((x0.reshape(1,-1),val[0],xT.reshape(1,-1)), axis=0)
            return xn, val[-1]
