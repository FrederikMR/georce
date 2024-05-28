#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:35:02 2024

@author: fmry
"""

#%% Sources

#%% Modules

from jaxman.setup import *
from jaxman.riemannian import RiemannianManifold
from .inverse_matrix import InvMat
from .line_search import LineSearch

#%% Geodesic Control

class GeodesicControl(object):
    def __init__(self,
                 M:RiemannianManifold,
                 tau:float=1.0,
                 T:float=1.0,
                 N:int=12,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 inv_method:str="Naive",
                 )->None:
        
        self.M = M
        self.tau = tau
        self.T = T
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        self.inv_fun = InvMat(ginv_fun = self.M.Ginv, method=inv_method)
            
        obj_fun = lambda x, *args: self.M.energy(x)
        update_fun = lambda x,alpha, *args: jnp.concatenate((x[0].reshape(1,-1),
                                                             x[0]+jnp.cumsum(alpha*args[0]+(1.-alpha)*args[1], axis=0),
                                                             x[-1].reshape(1,-1)), axis=0)
        self.LineSearch = LineSearch(obj_fun, update_fun,
                                     alpha_init=tau, decay=0.95,
                                     method='Soft')
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem"
    
    def unconstrained_opt(self, gn:Array, gn_inv:Array, x0:Array,xT:Array)->Array:
        
        g_cumsum = jnp.cumsum(gn[::-1], axis=0)
        ginv_sum = jnp.sum(gn_inv, axis=0)
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', gn_inv[:-1], g_cumsum[::-1]), axis=0)+2.0*self.diff
        lhs = -jnp.linalg.inv(ginv_sum)
        muT = jnp.einsum('ij,j->i', lhs, rhs)
        #muT = jnp.linalg.solve(ginv_sum, rhs)
        mun = jnp.concatenate((muT+g_cumsum[::-1], muT.reshape(1,-1)), axis=0)
        
        return mun
    
    def __call__(self, x0:Array, xT:Array, method="While")->Array:
        
        def control_for(carry, idx):
            
            xn, un = carry
            gn = jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
            gn_inv = self.inv_fun(xn[:-1])
            
            mun = self.unconstrained_opt(gn, gn_inv, x0, xT)
            
            un_hat = -0.5*jnp.einsum('tij,tj->ti', gn_inv, mun)
            tau = self.LineSearch(xn, un_hat, un)

            un = tau*un_hat+(1-tau)*un
            xn = jnp.concatenate((x0.reshape(1,-1), (x0+jnp.cumsum(un, axis=0))[:-1], xT.reshape(1,-1)), axis=0)
            
            return ((xn, un),)*2
        
        def control_while(carry):
            
            xn, un, gn, gn_inv, idx = carry
            
            mun = self.unconstrained_opt(gn, gn_inv, x0, xT)
            
            un_hat = -0.5*jnp.einsum('tij,tj->ti', gn_inv, mun)
            tau = self.LineSearch(xn, un_hat, un)

            un = tau*un_hat+(1-tau)*un
            xn = jnp.concatenate((x0.reshape(1,-1), (x0+jnp.cumsum(un, axis=0))[:-1], xT.reshape(1,-1)), axis=0)

            gn = jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
            gn_inv = self.inv_fun(xn[:-1])
            
            return (xn, un, gn, gn_inv, idx+1)
        
        self.diff = xT-x0
        
        t_grid = jnp.linspace(0,self.T,self.N+1).reshape(-1,1)
        #grad_fun = lambda x: jacfwd(lambda y: self.M.energy(jnp.concatenate((x0.reshape(1,-1),
        #                                                                     y,
        #                                                                     xT.reshape(1,-1)), axis=0)))(x)
        grad_fun = lambda x: self.M.Denergy(jnp.concatenate((x0.reshape(1,-1),
                                                             x,
                                                             xT.reshape(1,-1)), axis=0))
        
        xn = (x0+t_grid*self.diff).astype(jnp.float64)
        un = (jnp.ones_like(xn)[:-1]*self.diff/self.N).astype(jnp.float64)
        
        if method == "For":
            val, carry = lax.scan(control_for, init=(xn,un), xs=jnp.linspace(0,1,self.max_iter))
            return val, carry #(xn,un,gn,gn_inv),(xn,un,gn,gn_inv)
        else:
            gn = jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
            gn_inv = self.inv_fun(xn[:-1])
            
            cond_fun = lambda val: (jnp.linalg.norm(grad_fun(val[0][1:-1]))>self.tol)*(val[-1]<self.max_iter)
            val = lax.while_loop(cond_fun, control_while, init_val=(xn, un, gn, gn_inv, 0))
        
            return val[0], val[-1]
        
        
        
        
        
    