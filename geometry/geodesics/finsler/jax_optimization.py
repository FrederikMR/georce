#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.manifolds.finsler import FinslerManifold

#%% Gradient Descent Estimation of Geodesics

class JAXOptimization(ABC):
    def __init__(self,
                 M:FinslerManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lr_rate:float=1.0,
                 optimizer:Callable=None,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 obj_method:str="finsler",
                 )->None:
        
        self.M = M
        self.T = T
        self.max_iter = max_iter
        self.tol = tol
        
        if optimizer is None:
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr_rate)
        else:
            self.opt_init, self.opt_update, self.get_params = optimizer(lr_rate)
            
        if not (obj_method in ['tensor', 'finsler']):
                raise ValueError(f"The obj_method should be either tensor or finsler. Not {obj_method}.")
        else:
            if obj_method == 'tensor':
                self.obj_fun = M.g
            else:
                self.obj_fun = lambda z,u: M.F(z,u)**2
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
            
        self.z0 = None
        self.zT = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def energy(self, 
               zt:Array, 
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
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, Array, int],
                 )->Array:
        
        zt, grad, opt_state, idx = carry

        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array, Array, Array, int],
                   )->Array:
        
        zt, grad, opt_state, idx = carry
        
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)

        grad = self.Denergy(zt)
        
        return (zt, grad, opt_state, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array, Array],
                 idx:int,
                 )->Array:
        
        zt, opt_state = carry
        
        grad = self.Denergy(zt)
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)
        
        return ((zt, opt_state),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        zt = self.init_fun(z0,zT,self.T)
        
        self.z0 = z0
        self.zT = zT

        opt_state = self.opt_init(zt)
        
        if step == "while":
            grad = self.Denergy(zt)
        
            zt, grad, _, idx = lax.while_loop(self.cond_fun, 
                                              self.while_step,
                                              init_val=(zt, grad, opt_state, 0)
                                              )
        
            zt = jnp.vstack((z0, zt, zT))
        elif step == "for":
            _, val = lax.scan(self.for_step,
                              init=(zt, opt_state),
                              xs = jnp.ones(self.max_iter),
                              )
            
            zt = val[0]
            
            grad = vmap(self.Denergy)(zt)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return zt, grad, idx