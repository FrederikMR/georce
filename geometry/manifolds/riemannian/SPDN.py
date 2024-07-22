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

from .manifold import RiemannianManifold

#%% Code

class SPDN(RiemannianManifold):
    def __init__(self,
                 N:int=2,
                 coordinates="stereographic",
                 )->None:
        
        self.N = N
        self.dim = N*(N+1)//2
        self.emb_dim = N*N
        
        super().__init__(G=None, f=self.f, invf = self.invf)
        
        return
    
    def __str__(self):
        return "SPDN(%d), dim %d" % (self.N,self.dim)
    
    def f(self,
          x:Array
          )->Array:
        
        l = jnp.zeros((self.N, self.N))
        l = l.at[jnp.triu_indices(self.N, k=0)].set(x)
        
        return l.T.dot(l).reshape(-1)
    
    def invf(self, 
             x:Array
             )->Array:
        
        P = x.reshape(self.N, self.N)
        l = jnp.linalg.cholesky(P).T
        
        l = l[jnp.triu_indices(self.N, k=0)]
        
        return l.reshape(-1)
    
    def Exp(self, 
            x:Array, 
            v:Array, 
            t:float=1.0
            )->Array:
        
        P = x.reshape(self.N,self.N)
        v = v.reshape(self.N,self.N)
        
        U,S,V = jnp.linalg.svd(P)
        P_phalf = jnp.dot(jnp.dot(U, jnp.diag(jnp.sqrt(S))), V)#jnp.linalg.cholesky(P)
        P_nhalf = jnp.linalg.inv(P_phalf)#jnp.linalg.inv(P_phalf)
        
        exp_val = jnp.dot(jnp.dot(P_nhalf, v), P_nhalf)
        exp_val = jscipy.linalg.expm(exp_val)
        
        P_exp = jnp.dot(jnp.dot(P_phalf, exp_val), P_phalf)
        P_exp = 0.5*(P_exp+P_exp.T) #For numerical stability
        
        return lax.select(jnp.linalg.det(P_exp)<1e-2  , P.reshape(-1), P_exp.reshape(-1))
    
    def Log(self, 
            x:Array, 
            y:Array
            )->Array:
        
        P = x.reshape(self.N,self.N)
        Q = y.reshape(self.N,self.N)
        
        U,S,V = jnp.linalg.svd(P)
        P_phalf = jnp.dot(jnp.dot(U, jnp.diag(jnp.sqrt(S))), V)#jnp.linalg.cholesky(P)
        P_nhalf = jnp.linalg.inv(P_phalf)#jnp.linalg.inv(P_phalf)
        
        jnp.dot(U,jnp.dot(jnp.diag(jnp.log(S)),V))
        
        log_val = jnp.matmul(jnp.matmul(P_nhalf, Q), P_nhalf)
        U,S,V = jnp.linalg.svd(log_val)
        log_val = jnp.dot(U,jnp.dot(jnp.diag(jnp.log(S)),V))
        
        w = jnp.matmul(jnp.matmul(P_phalf, log_val), P_phalf)
            
        return w
    
    def dist(self, 
             x:Array, 
             y:Array
             )->Array:
        
        x = self.f(x)
        y = self.f(y)
        
        P1 = x.reshape(self.N,self.N)
        P2 = y.reshape(self.N,self.N)
        
        U, S, Vh = jnp.linalg.svd(jnp.linalg.solve(P1, P2))
        
        return jnp.sqrt(jnp.sum(jnp.log(S)**2))
    
    def Geodesic(self,
                 x:Array,
                 y:Array,
                 t_grid:Array=None,
                 )->Array:
        
        if t_grid is None:
            t_grid = jnp.linspace(0.,1.,99, endpoint=False)
        
        x = self.f(x)
        y = self.f(y)
        
        v = self.Log(x,y)
        
        gamma = vmap(lambda t: self.Exp(x, v,t))(t_grid).reshape(-1, self.emb_dim)
        gamma = vmap(lambda g: self.invf(g))(gamma)
        
        
        gamma = gamma.reshape(-1, self.emb_dim)
        
        return jnp.vstack((x,gamma,y))
    
    
    
    
    