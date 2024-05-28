#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:54:30 2024

@author: fmry
"""

#%% Sources

#%% Modules

from setup import *

#%% Riemannian Manifold

class RiemannianManifold(ABC):
    def __init__(self,
                 G:Callable[[Array], Array]=None,
                 F:Callable[[Array], Array]=None,
                 invF:Callable[[Array],Array]=None,
                 )->None:
        
        self.F = F
        self.invF = invF
        if ((G is None) and (F is None)):
            raise ValueError("Both the metric, g, and chart, F, is not defined")
        elif ((G is not None) and (F is not None)):
            raise ValueError("Both the metric, g, and chart, F, is defined. Choose only one to define the metric")
        elif (G is None):
            self.G = lambda z: self.pull_back_metric(z)
        else:
            self.G = G
            
        return
        
    def __str__(self)->str:
        
        return "Riemannian Manifold base object"
    
    def JF(self,
           z:Array
           )->Array:
        
        if self.F is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            return jacfwd(self.F)(z)
        
    def pull_back_metric(self,
                         z:Array
                         )->Array:
        
        if self.F is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            JF = self.JF(z)
            return jnp.einsum('ik,il->kl', JF, JF)
    
    def DG(self,
           z:Array
           )->Array:

        return jacfwd(self.G)(z)
    
    def Ginv(self,
             z:Array
             )->Array:
        
        return jnp.linalg.inv(self.G(z))
    
    def christoffel_symbols(self,
                            z:Array
                            )->Array:
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        
        return 0.5*(jnp.einsum('im,kml->ikl',gsharpx,Dgx)
                   +jnp.einsum('im,lmk->ikl',gsharpx,Dgx)
                   -jnp.einsum('im,klm->ikl',gsharpx,Dgx))
    
    def euler_lagrange(self, 
                       z:Array, 
                       v:Array
                       )->Array:
        
        Gamma = self.Chris(z)

        dx1t = v
        dx2t = -jnp.einsum('ikl,k,l->i',Gamma,v,v)
        
        return jnp.hstack((dx1t,dx2t))
    
    def energy(self, 
               gamma:Array,
               )->Array:
        
        T = 1./len(gamma)
        dgamma = gamma[1:]-gamma[:-1]
        
        g = vmap(lambda g: self.G(g))(gamma)
        integrand = jnp.einsum('ti,tij,tj->t', dgamma, g[:-1], dgamma)
        
        return jnp.sum(integrand)/T
    
    def length(self,
               gamma:Array,
               )->Array:
        
        dgamma = gamma[1:]-gamma[:-1]
        
        g = vmap(lambda g: self.G(g))(gamma)
        integrand = jnp.sqrt(jnp.einsum('ti,tij,tj->t', dgamma, g[:-1], dgamma))
            
        return jnp.sum(integrand)
    