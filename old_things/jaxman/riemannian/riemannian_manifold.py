#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:10:07 2024

@author: fmry
"""

#%% Modules

from jaxman.setup import *

#%% Riemannian Manifold

class RiemannianManifold(object):
    def __init__(self,
                 g:Callable[[Array], Array] = None,
                 F:Callable[[Array], Array] = None,
                 geodesic_opt:str = "Control"
                 )->None:
        
        self.F = F
        if ((g is None) and (F is None)):
            raise ValueError("Both the metric, g, and chart, F, is not defined")
        elif ((g is not None) and (F is not None)):
            raise ValueError("Both the metric, g, and chart, F, is defined. Choose only one to define the metric")
        elif (g is None):
            self.g = lambda z: self.G(z)
        else:
            self.G = g

        return
            
    def __str__(self)->str:
        
        return "Riemannian Manifold object"
        
    def JF(self,z:Array)->Array:
        
        if z.ndim == 1:
            return jacfwd(self.F)(z)
        else:
            return vmap(jacfwd(self.F))(z)
        
    def G(self,z:Array)->Array:
        
        JF = self.JF(z)
        
        return jnp.einsum('...ik,...il->...kl', JF, JF)
    
    def DG(self,z:Array)->Array:
        
        dg = jacfwd(self.G)
        
        if z.ndim == 1:
            dg(z)
        else:
            return vmap(lambda z: dg(z))(z)
    
    def Ginv(self,z:Array)->Array:
        
        return jnp.linalg.inv(self.G(z))
    
    def Chris(self,z:Array)->Array:
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        
        return 0.5*(jnp.einsum('...im,...kml->...ikl',gsharpx,Dgx)
                   +jnp.einsum('...im,...lmk->...ikl',gsharpx,Dgx)
                   -jnp.einsum('...im,...klm->...ikl',gsharpx,Dgx))
    
    def euler_lagrange(self, z, v):
        
        z = z.T
        v = v.T
        
        Gamma = self.Chris(z)

        dx1t = v
        dx2t = -jnp.einsum('...ikl,...k,...l->...i',Gamma,v,v)
        
        return jnp.hstack((dx1t,dx2t)).T
    
    def trapez_rule(self, integrand:Array, dt_grid)->Array:

        return jnp.sum(0.5*(integrand[:-1]+integrand[1:])*dt_grid)
        
    def energy(self, gamma:Array, g:Array=None)->Array:

        dgamma = gamma[1:]-gamma[:-1]
        g = self.G(gamma[:-1])
        t_grid = jnp.linspace(0.0,1.0, len(dgamma))
        dt_grid = jnp.diff(t_grid).reshape(-1,1)
            
        integrand = jnp.einsum('ti,tij,tj->t', dgamma, g, dgamma)
        
        return jnp.sum(integrand)
        
        #return jnp.sum(0.5*(integrand[1:]+integrand[:-1])*dt_grid)
    
    def Denergy(self, gamma:Array)->Array:
        
        dg = self.DG(gamma[:-1])
        g = self.G(gamma[:-1])
        dgamma = gamma[1:]-gamma[:-1]
            
        integrand = jnp.einsum('ti,tijd,tj->td', dgamma[1:], dg[1:], dgamma[1:])
        integrand += 2.*(jnp.einsum('tij,tj->ti', g[:-1], dgamma[:-1])-jnp.einsum('tij,tj->ti', g[1:], dgamma[1:]))
        
        return integrand
        
    def length(self, gamma:Array, g:Array=None)->Array:
        
        dgamma = gamma[1:]-gamma[:-1]
        if g is None:
            g = self.G(gamma[1:])
        t_grid = jnp.linspace(0.0,1.0, len(dgamma))
        dt_grid = jnp.diff(t_grid).reshape(-1,1)
            
        integrand = jnp.sqrt(jnp.einsum('ti,tij,tj->t', dgamma, g, dgamma))
        
        return jnp.sum(0.5*(integrand[1:]+integrand[:-1])*dt_grid)
    
    
    