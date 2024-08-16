#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from jax import Array
from jax import vmap, grad, jacfwd
from jax import lax

import jax.numpy as jnp

from abc import ABC
from typing import Callable, Tuple, Dict 
    
#%% Backtracking Line Search

class Backtracking(ABC):
    def __init__(self,
                 obj_fun:Callable[[Array,...], Array],
                 update_fun:Callable[[Array, Array,...], Array],
                 grad_fun:Callable[[Array, ...], Array],
                 criterion:str="armijo",
                 alpha:float=1.0,
                 rho:float=0.9,
                 c:float=0.25,
                 c1:float=0.90,
                 c2:float=0.1,
                 max_iter:int=100,
                 )->None:
        #https://optimization.cbe.cornell.edu/index.php?title=Line_search_methods
        
        self.obj_fun = obj_fun
        self.update_fun = update_fun
        self.grad_fun = grad_fun
        
        self.alpha = alpha
        self.rho = rho
        self.c = c
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        
        if criterion == "fixed":
            self.condition = self.fixed_condition
        elif criterion == "naive":
            self.condition = self.naive_condition
        elif criterion == "armijo":
            self.condition = self.armijo_condition
        elif criterion == "curvature":
            self.condition = self.curvature_condition
        elif criterion == "strong_curvature":
            self.condition = self.strong_curvature_condition
        elif criterion == "wolfe":
            self.condition = self.wolfe_condition
        elif criterion == "strong_wolfe":
            self.condition = self.strong_wolfe_condition
        elif criterion == "goldstein":
            self.condition = self.goldstein_condition
        else:
            raise ValueError("Invalid criterion for line search")
        
        self.x = None
        self.obj0 = None
        
        return
    
    def fixed_condition(self, x_new:Array, obj:Array, alpha:Array, *args)->bool:
        
        return False
    
    def naive_condition(self, x_new:Array, obj:Array, alpha:Array, *args)->bool:
        
        return obj>self.obj0
    
    def armijo_condition(self, x_new:Array, obj:Array, alpha:Array, *args)->bool:
        
        val1 = self.obj0+self.c1*alpha*jnp.dot(self.pk, self.grad0)
        
        return obj>val1
    
    def curvature_condition(self, x_new:Array, obj:Array, alpha:Array, *args)->bool:
        
        grad_val = self.grad_fun(x_new, *args)
        val0 = jnp.dot(grad_val, self.pk)
        val1 = self.c2*jnp.dot(self.pk, self.grad0)
        
        return val0 < val1
    
    def strong_curvature_condition(self, x_new:Array, obj:Array, alpha:Array, *args)->bool:
        
        grad_val = self.grad_fun(x_new, *args)
        val0 = jnp.dot(grad_val, self.pk)
        val1 = self.c2*jnp.dot(self.pk, self.grad0)
        
        return jnp.abs(val0) > jnp.abs(val1)
    
    def wolfe_condition(self, x_new:Array, obj:Array, alpha:Array, *args)->bool:
        
        armijo = self.armijo_condition(x_new, obj, alpha, *args)
        curvature = self.curvature_condition(x_new, obj, alpha, *args)
        
        return armijo & curvature
    
    def strong_wolfe_condition(self, x_new:Array, obj:Array, alpha:Array, *args)->bool:
        
        armijo = self.armijo_condition(x_new, obj, alpha, *args)
        curvature = self.strong_curvature_condition(x_new, obj, alpha, *args)
            
        return armijo & curvature
    
    def goldstein_condition(self, x_new:Array, obj:Array, alpha:Array, *args)->bool:
        
        val0 = self.obj0 + (1-self.c)*alpha*jnp.dot(self.grad0, self.pk)
        val1 = self.obj0 + self.c*alpha*jnp.dot(self.grad0, self.pk)
        
        bool1 = val0>obj
        bool2 = val1<obj
        
        return bool1 & bool2
    
    def cond_fun(self, 
                 carry:Tuple[Array, int],
                 )->Array:
        
        alpha, idx, *args = carry
        
        x_new = self.update_fun(self.x, alpha, *args)
        obj = self.obj_fun(x_new, *args)
        
        bool_val = self.condition(x_new, obj, alpha, *args)
        
        return (bool_val) & (idx < self.max_iter)
    
    def update_alpha(self,
                     carry:Tuple[Array, int]
                     )->Array:
        
        alpha, idx, *_ = carry
        
        return (self.rho*alpha, idx+1, *_)
    
    def __call__(self, 
                 x:Array,
                 *args,
                 )->Array:
        
        self.x = x
        self.obj0 = self.obj_fun(x,*args)
        grad_val = self.grad_fun(x,*args)
        self.pk = -grad_val
        self.grad0 = grad_val
        
        alpha, *_ = lax.while_loop(self.cond_fun,
                                   self.update_alpha,
                                   init_val = (self.alpha, 0, *args)
                                   )
        
        return alpha