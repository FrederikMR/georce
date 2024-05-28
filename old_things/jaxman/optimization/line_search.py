#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:35:02 2024

@author: fmry
"""

#%% Sources

#%% Modules

from jaxman.setup import *

#%% Inverse Matrix Computations

class LineSearch(object):
    def __init__(self,
                 obj_fun:Callable[[Array], Array],
                 update_fun:Callable[[Array,Array],Array],
                 alpha_init:float=1.0,
                 decay:float=0.90,
                 method:str = 'Soft',
                 max_iter:int=10,
                 )->None:

        if not(method in ['Soft', 'Hard']):
            raise ValueError(f"Method, {method}, should be either: \n\t-Soft \n\t-Hard")
            
        self.obj_fun = obj_fun
        self.update_fun = update_fun
        self.alpha = alpha_init
        self.decay = decay
        self.method = method
        self.max_iter = max_iter
        
        return
    
    def __str__(self)->str:
        
        return "Object Computing Inverse Matrix"
    
    def __call__(self, x:Array, *args)->Array:
        
        def update_step(carry):
            
            alpha, idx = carry
            
            return (self.decay*alpha, idx+1)

        obj = self.obj_fun(x,*args)
        alpha = lax.while_loop(lambda a: lax.cond((self.obj_fun(self.update_fun(x, a[0], *args), *args)>obj)*(a[1]<self.max_iter),
                                                  lambda: True,
                                                  lambda : False,
                                                  ),
                               update_step, 
                               init_val=(self.alpha,0)
                               )
        self.alpha = alpha[0]

        return alpha[0]