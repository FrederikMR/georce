#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.manifolds.riemannian import RiemannianManifold

#%% Gradient Descent Estimation of Geodesics

class ScipyBVP(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 method:str='RK45',
                 )->None:
        
        if method not in['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']:
            raise ValueError(f"Method, {method}, is not defined in scipy_ivp.")

        self.M = M
        self.T = T
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        
        self.save_zt = []
        
        self.dim = None
        self.z0 = None
        self.zT = None
        self.G0 = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Scipy Optimizers"
    
    def dif_fun(self,
                t:Array,
                y:Array,
                )->Array:

        x = y[:self.M.dim]
        v = y[self.M.dim:]
        
        return self.M.geodesic_equation(x,v)
    
    def obj_fun(self, 
                v:Array, 
                )->Array:
        
        sol = solve_ivp(self.dif_fun, self.t_span, jnp.hstack((self.z0, v)),
                        method=self.method, t_eval=self.t_eval)
        
        return jnp.sum((sol.y[:self.dim,-1]-self.zT)**2)
    
    def callback(self,
                 zt:Array
                 )->Array:
        
        self.save_zt.append(zt.reshape(-1, self.dim))
        
        return
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 )->Array:
        
        self.dim = len(z0)
        
        self.z0 = z0
        self.zT = zT
        self.t_span = [0., 1.]
        self.t_eval = jnp.linspace(0.,1.,self.T+1, endpoint=True)
        
        #if self.method == "BFGS":
        #    min_fun = jminimize
        #else:
        min_fun = minimize

        res = min_fun(fun = self.obj_fun, 
                      x0=jnp.ones_like(self.z0), 
                      method='BFGS', 
                      tol=self.tol,
                      options={'maxiter': self.max_iter}
                      )
    
        v = res.x
        zt = solve_ivp(self.dif_fun, self.t_span, jnp.hstack((self.z0, v)),
                        method=self.method, t_eval=self.t_eval).y[:self.M.dim].T

        grad =  res.jac.reshape(-1,self.dim)
        idx = res.nit
        
        zt = jnp.array(zt)
        zt = zt.at[0].set(self.z0)
        zt = zt.at[-1].set(self.zT)
        
        return zt, grad, idx
    
    
    