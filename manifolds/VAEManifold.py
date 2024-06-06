#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

from setup import *

####################

from .riemannian import RiemannianManifold

#%% Code

class VAEManifold(RiemannianManifold):
    def __init__(self,
                 dim,
                 emb_dim,
                 encoder:Callable[[Array], Array],
                 decoder:Callable[[Array], Array],
                 )->None:

        self.dim = dim
        self.emb_dim = emb_dim
        super().__init__(F=decoder, invF=encoder)
        
        return