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

class InvMat(object):
    def __init__(self,
                 ginv_fun:Callable[[Array], Array],
                 ginv_init:Array=None,
                 method:str = 'Naive',
                 )->None:

        if method == "Naive":
            self.method = method
        else:
            raise ValueError(f"Method {self.method} is not implemented. Choose: Naive")
        self.ginv_init = ginv_init
        self.ginv_fun = ginv_fun
        
        return
    
    def __str__(self)->str:
        
        return "Object Computing Inverse Matrix"
    
    def __call__(self, 
                 z:Array
                 )->Array:
        
        if self.method == "Naive":
            return self.ginv_fun(z)