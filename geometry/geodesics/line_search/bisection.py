#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:17:25 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

#%% Bisection method

class Bisection(ABC):
    def __init__(self,
                 obj_fun:Callable[[Array, ...], Array],
                 update_fun:Callable[[Array, Array,...], Array],
                 tol:float=1e-4,
                 max_iter:int=100,
                 alpha_min:float=0.0,
                 alpha_max:float=1.0,
                 )->None:
        
        self.obj_fun = obj_fun
        self.update_fun = update_fun
        self.tol = tol
        self.max_iter = max_iter
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        self.x = None
        
        return
    
    def cond_fun(self, 
                 carry:Tuple[Array, int],
                 )->Array:
        
        alpha, alpha0, alpha1, obj0, obj1, idx, *args = carry

        return (jnp.abs(obj1-obj0)>self.tol) & (idx < self.max_iter)
    
    def update_alpha(self,
                     carry:Tuple[Array, int]
                     )->Array:
        
        alpha, alpha0, alpha1, obj0, obj1, idx, *_ = carry
        
        limits = lax.cond(obj0<obj1,
                          lambda alpha0,alpha,alpha1: (alpha0, alpha),
                          lambda alpha0,alpha,alpha1: (alpha, alpha1),
                          *(alpha0, alpha, alpha1),
                          )
        
        alpha0, alpha1 = limits[0], limits[1]
        alpha = (alpha0+alpha1)*0.5
        
        obj0 = self.obj_fun(self.update_fun(self.x, alpha0, *_))
        obj1 = self.obj_fun(self.update_fun(self.x, alpha1, *_))
        
        return (alpha, alpha0, alpha1, obj0, obj1, idx, *_)
    
    def __call__(self, 
                 x:Array,
                 *args,
                 )->Array:
        
        self.x = x
        obj0 = self.obj_fun(x,self.alpha_min, *args)
        obj1 = self.obj_fun(self.update_fun(self.x, self.alpha_max, *args))
        alpha = 0.5*(self.alpha_min+self.alpha_max)
        
        alpha, *_ = lax.while_loop(self.cond_fun,
                                   self.update_alpha,
                                   init_val = (alpha, self.alpha_min, self.alpha_max, obj0, obj1, 0, *args)
                                   )
        
        return alpha