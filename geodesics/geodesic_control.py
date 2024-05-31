#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from setup import *

from manifolds import RiemannianManifold
from .line_search import SoftLineSearch

#%% Gradient Descent Estimation of Geodesics

class GC_LineSearch(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lr_rate:float=1.0,
                 T:int=10,
                 decay_rate:float=0.95,
                 tol:float=1e-8,
                 max_iter:int=1000,
                 line_search_iter:int=100,
                 )->None:
        
        self.M = M
        self.lr_rate = lr_rate
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        
        self.decay_rate = decay_rate
        self.line_search_iter = line_search_iter
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   self.T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem"
    
    def energy(self, 
               zt:Array,
               *args,
               )->Array:
        
        term1 = zt[0]-self.z0
        val1 = jnp.einsum('i,ij,j->', term1, self.G0, term1)
        
        term2 = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.M.G(z))(zt)
        val2 = jnp.einsum('ti,tij,tj->t', term2, Gt[:-1], term2)
        
        term3 = self.zT-zt[-1]
        val3 = jnp.einsum('i,ij,j->', term3, Gt[-1], term3)
        
        return val1+jnp.sum(val2)+val3
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        return grad(lambda z: self.energy(z))(zt)
    
    def inner_product(self,
                      zt:Array,
                      ut:Array,
                      )->Array:
        
        Gt = vmap(lambda z: self.M.G(z))(zt)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', ut, Gt, ut))
    
    def gt(self,
           zt:Array,
           ut:Array,
           )->Array:
        
        return grad(self.inner_product)(zt,ut)
    
    def update_xt(self,
                  zt:Array,
                  alpha:Array,
                  ut_hat:Array,
                  ut:Array,
                  )->Array:
        
        return self.z0+jnp.cumsum(alpha*ut_hat[:-1]+(1-alpha)*ut[:-1], axis=0)
    
    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        zt, ut, gt, gt_inv, grad, idx = carry
        
        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def control_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        zt, ut, gt, gt_inv, grad, idx = carry
        
        mut = self.unconstrained_opt(gt, gt_inv)

        ut_hat = -0.5*jnp.einsum('tij,tj->ti', gt_inv, mut)
        tau = self.line_search(zt, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)

        gt = self.gt(zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.vstack((self.Ginv0, vmap(lambda z: self.M.Ginv(z))(zt)))
        grad = self.Denergy(zt)
        
        return (zt, ut, gt, gt_inv, grad, idx+1)
    
    def unconstrained_opt(self, gt:Array, gt_inv:Array)->Array:
        
        g_cumsum = jnp.cumsum(gt[::-1], axis=0)[::-1]
        ginv_sum = jnp.sum(gt_inv, axis=0)
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', gt_inv[:-1], g_cumsum), axis=0)+2.0*self.diff
        #lhs = -jnp.linalg.inv(ginv_sum)
        #muT = jnp.einsum('ij,j->i', lhs, rhs)
        muT = -jnp.linalg.solve(ginv_sum, rhs)
        mut = jnp.vstack((muT+g_cumsum, muT))
        
        return mut
    
    def __call__(self, 
                 z0:Array,
                 zT:Array
                 )->Array:
        
        dtype = z0.dtype
        self.dim = len(z0)
        
        zt = self.init_fun(z0,zT,self.T)
        
        self.line_search = SoftLineSearch(obj_fun=self.energy,
                                          update_fun=self.update_xt,
                                          alpha=self.lr_rate,
                                          decay_rate=self.decay_rate,
                                          max_iter=self.line_search_iter
                                          )
        
        self.z0 = z0
        self.G0 = self.M.G(z0)
        self.Ginv0 = jnp.linalg.inv(self.G0).reshape(1,self.dim,self.dim)
        self.zT = zT
        self.diff = zT-z0
        
        
        ut = jnp.ones((self.T, self.dim), dtype=dtype)*self.diff/self.T
        
        gt = self.gt(zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.vstack((self.Ginv0, vmap(lambda z: self.M.Ginv(z))(zt)))
        grad = self.Denergy(zt)
            
        zt, _, _, _, grad, idx = lax.while_loop(self.cond_fun, 
                                                self.control_step, 
                                                init_val=(zt, ut, gt, gt_inv, grad, 0))
        
        zt = jnp.vstack((z0, zt, zT))
        
        return zt, grad, idx

        