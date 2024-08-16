#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from .setup import *

from .backtracking import Backtracking

#%% GEORCE Estimation of Geodesics

class GEORCE_F(ABC):
    def __init__(self,
                 F:Callable[[Array,Array],Array],
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {},
                 )->None:
        
        self.F = F
        self.obj_fun = lambda z,u: self.F(z,u)**2
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem"
    
    def G(self, 
          z:Array, 
          v:Array
          )->Array:
        
        return 0.5*jacfwd(lambda v1: grad(lambda v2: self.F(z,v2)**2)(v1))(v)
    
    def Ginv(self, 
             z:Array, 
             v:Array
             )->Array:
        
        return jnp.linalg.inv(self.G(z,v))
    
    def energy(self, 
               zt:Array, 
               *args
               )->Array:
        
        term1 = zt[0]-self.z0
        val1 = self.obj_fun(self.z0, term1)
        
        term2 = zt[1:]-zt[:-1]
        val2 = vmap(lambda x,v: self.obj_fun(x,v))(zt[:-1], term2)
        
        term3 = self.zT-zt[-1]
        val3 = self.obj_fun(zt[-1], term3)
        
        return val1+jnp.sum(val2)+val3
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        return grad(lambda z: self.energy(z))(zt)
    
    def inner_product(self,
                      zt:Array,
                      ut:Array,
                      )->Array:
        
        Gt = vmap(self.G)(zt,ut)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', ut, Gt, ut))
    
    def inner_product_h(self,
                        zt:Array,
                        u0:Array,
                        ut:Array,
                        )->Array:
        
        Gt = vmap(self.G)(zt,ut)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', u0, Gt, u0))
    
    def gt(self,
           zt:Array,
           ut:Array,
           )->Array:
        
        return grad(self.inner_product, argnums=0)(zt,ut)
    
    def ht(self,
           zt:Array,
           ut:Array,
           )->Array:

        return grad(self.inner_product_h, argnums=2)(zt,ut,ut)
    
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
        
        zt, ut, ht, gt, gt_inv, grad, idx = carry
        
        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        zt, ut, ht, gt, gt_inv, grad, idx = carry
        
        mut = self.unconstrained_opt(ht, gt, gt_inv)

        ut_hat = -0.5*jnp.einsum('tij,tj->ti', gt_inv, mut)
        tau = self.line_search(zt, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)

        ht = self.ht(zt,ut[:-1])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt = self.gt(zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.vstack((self.Ginv(self.z0, ut[0]).reshape(-1,self.dim,self.dim), 
                             vmap(self.Ginv)(zt,ut[1:])))
        grad = self.Denergy(zt)
        
        return (zt, ut, ht, gt, gt_inv, grad, idx+1)
    
    def unconstrained_opt(self, ht:Array, gt:Array, gt_inv:Array)->Array:
        
        g_cumsum = jnp.cumsum(gt[::-1], axis=0)[::-1]
        ginv_sum = jnp.sum(gt_inv, axis=0)
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', gt_inv[:-1], g_cumsum+ht), axis=0)+2.0*self.diff
        #lhs = -jnp.linalg.inv(ginv_sum)
        #muT = jnp.einsum('ij,j->i', lhs, rhs)
        muT = -jnp.linalg.solve(ginv_sum, rhs)
        mut = jnp.vstack((muT+g_cumsum+ht, muT))
        
        return mut
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 )->Array:
        
        dtype = z0.dtype
        self.dim = len(z0)
        
        zt = self.init_fun(z0,zT,self.T)
        
        self.line_search = Backtracking(obj_fun=self.energy,
                                        update_fun=self.update_xt,
                                        grad_fun = lambda z,*args: self.Denergy(z).reshape(-1),
                                        **self.line_search_params,
                                        )
        
        self.diff = zT-z0
        ut = jnp.ones((self.T, self.dim), dtype=dtype)*self.diff/self.T
        
        self.z0 = z0
        self.zT = zT
        
        gt = self.gt(zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        ht = self.ht(zt,ut[:-1])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.vstack((self.Ginv(self.z0, ut[0]).reshape(-1,self.dim,self.dim), 
                             vmap(self.Ginv)(zt,ut[1:])))
        grad = self.Denergy(zt)
        
        zt, *_ = lax.while_loop(self.cond_fun, 
                                self.while_step, 
                                init_val=(zt, ut, ht, gt, gt_inv, grad, 0),
                                )
        
        zt = jnp.vstack((z0, zt, zT))
            
        return zt

        