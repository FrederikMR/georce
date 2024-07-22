#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

####################

from geometry.manifolds.riemannian import RiemannianManifold
from .manifold import FinslerManifold

#%% Code

class RiemannianNavigation(FinslerManifold):
    def __init__(self,
                 RM:RiemannianManifold,
                 force_fun:Callable[[Array], Array],
                 v0:float,
                 )->None:
        
        self.RM = RM
        self.force_fun = force_fun
        self.v0 = v0

        self.dim = RM.dim
        self.emb_dim = RM.emb_dim
        super().__init__(F=self.metric, G=self.fundamental_tensor, f=self.RM.f, invf=self.RM.invf)
        
        return
    
    def __str__(self)->str:
        
        return f"Randers manifold of dimension {self.dim} for manifold of type: \n\t-{self.RM.__str__()}"
    
    def fundamental_tensor(self,
                           z:Array,
                           v:Array,
                           )->Array:
        
        g = self.RM.G(z)
        force = self.force_fun(z)
        
        lam = 1./((self.v0**2)-jnp.einsum('ij,i,j->', g, force, force))
        f = jnp.dot(g, force)
        
        a = g*lam+jnp.einsum('i,j->ij', f, f)*(lam**2)
        b = -f*lam
        
        inner = jnp.sqrt(jnp.einsum('ij,i,j->', a, v, v))
        l = jnp.dot(a, v)/inner
        
        gv = (1.0+jnp.dot(b, v)/inner)*(a-jnp.einsum('i,j->ij', l, l))\
            +jnp.einsum('i,j->ij', b+l, b+l)
        
        return 0.5*gv
    
    def metric(self,
               z:Array,
               v:Array,
               )->Array:
        
        
        g = self.RM.G(z)
        force = self.force_fun(z)
        
        lam = 1./((self.v0**2)-jnp.einsum('ij,i,j->', g, force, force))
        f = jnp.dot(g, force)
        
        a = g*lam+jnp.einsum('i,j->ij', f, f)*(lam**2)
        b = -f*lam
        
        term1 = jnp.einsum('ij,i,j->', a, v, v)
        term2 = jnp.dot(b, v)
        
        return jnp.sqrt(term1)+term2