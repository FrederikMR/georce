#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from setup import *

#%% Soft Line Search

class SoftLineSearch(ABC):
    def __init__(self,
                 obj_fun:Callable[[Array,...], Array],
                 update_fun:Callable[[Array, Array,...], Array],
                 alpha:Array,
                 decay_rate:float=.95,
                 max_iter:int=100,
                 )->None:
        
        self.obj_fun = obj_fun
        self.update_fun = update_fun
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.max_iter = max_iter
        
        self.x = None
        self.obj0 = None
        
        return
    
    def cond_fun(self, 
                 carry:Tuple[Array, int],
                 )->Array:
        
        alpha, idx, *args = carry
        
        obj = self.obj_fun(self.update_fun(self.x, alpha, *args))
        
        return (obj>self.obj0) & (idx < self.max_iter)
    
    def update_alpha(self,
                     carry:Tuple[Array, int]
                     )->Array:
        
        alpha, idx, *_ = carry
        
        return (self.decay_rate*alpha, idx+1, *_)
    
    def __call__(self, 
                 x:Array,
                 *args,
                 )->Array:
        
        self.x = x
        self.obj0 = self.obj_fun(x,*args)
        
        alpha, *_ = lax.while_loop(self.cond_fun,
                                   self.update_alpha,
                                   init_val = (self.alpha, 0, *args)
                                   )
        
        return alpha
    
#%% Exact Line Search

class ExactLineSearch(ABC):
    def __init__(self,
                 obj_fun:Callable[[Array,...], Array],
                 update_fun:Callable[[Array, Array,...], Array],
                 alpha:Array,
                 decay_rate:float=.95,
                 max_iter:int=100,
                 )->None:
        
        self.obj_fun = obj_fun
        self.update_fun = update_fun
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.max_iter = max_iter
        
        self.x = None
        self.obj0 = None
        
        return
    
    def cond_fun(self, 
                 carry:Tuple[Array, int],
                 )->Array:
        
        alpha, idx, *args = carry
        
        obj = self.obj_fun(self.update_fun(self.x, alpha, *args))
        
        return (obj>self.obj0) & (idx < self.max_iter)
    
    def update_alpha(self,
                     carry:Tuple[Array, int]
                     )->Array:
        
        alpha, idx, *_ = carry
        
        return (self.decay_rate*alpha, idx+1, *_)
    
    def __call__(self, 
                 x:Array,
                 *args,
                 )->Array:
        
        self.x = x
        self.obj0 = self.obj_fun(x,*args)
        
        alpha, *_ = lax.while_loop(self.cond_fun,
                                   self.update_alpha,
                                   init_val = (self.alpha, 0, *args)
                                   )
        
        return alpha