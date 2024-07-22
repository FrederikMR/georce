#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 00:42:04 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

####################

from .manifold import FinslerManifold
from geometry.regression import GPRegression

#%% Expected Length for Gaussian Process

class GPFinsler(FinslerManifold, GPRegression):
    def __init__(self,
                 X_training:Array,
                 y_training:Array,
                 mu_fun:Callable[[Array, ...], Array] = None,
                 k_fun:Callable[[Array, Array, ...], Array] = None,
                 optimize_hyper_parameters:bool=False,
                 sigma:float=1.0,
                 lr_rate:float=0.01,
                 optimizer:Callable=None,
                 max_iter:int=100,
                 delta:float=1e-10,
                 kernel_params:Array = jnp.array([1.0, 1.0]),
                 Dk_fun:Callable[[Array, Array, ...], Array] = None,
                 DDk_fun:Callable[[Array, Array, ...], Array] = None,
                 DDDk_fun:Callable[[Array, Array, ...], Array] = None,
                 DDDDk_fun:Callable[[Array, Array, ...], Array] = None,
                 Dmu_fun:Callable[[Array, ...], Array] = None,
                 DDmu_fun:Callable[[Array, ...], Array] = None,
                 DDDmu_fun:Callable[[Array, ...], Array] = None,
                 )->None:
        
        GPRegression.__init__(self, 
                              X_training=X_training,
                              y_training=y_training,
                              mu_fun = mu_fun,
                              k_fun = k_fun,
                              optimize_hyper_parameters = optimize_hyper_parameters,
                              sigma = sigma,
                              lr_rate = lr_rate,
                              optimizer=optimizer,
                              max_iter=max_iter,
                              delta=delta,
                              kernel_params=kernel_params
                              )
        
        if mu_fun is None:
            self.Dmu_fun = lambda x: jnp.zeros((x.shape[-1], self.dim))
            self.DDmu_fun = lambda x: jnp.zeros((x.shape[-1], self.dim, self.dim))
            self.DDDmu_fun = lambda x: jnp.zeros((x.shape[-1], self.dim, self.dim, self.dim))
        else:
            if Dmu_fun is None:
                self.Dmu_fun = vmap(lambda x: grad(self.mu_fun))
            else:
                self.Dmu_fun = Dmu_fun
            if DDmu_fun is None:
                self.DDmu_fun = vmap(lambda x: jacfwd(self.Dmu_fun))
            else:
                self.DDmu_fun = DDmu_fun
            if DDDmu_fun is None:
                self.DDDmu_fun = vmap(lambda x: jacfwd(self.DDmu_fun))
            else:
                self.DDDmu_fun = DDDmu_fun

        if Dk_fun is None:
            self.Dk_fun = grad(self.k_fun, argnums=0)
        if DDk_fun is None:
            self.DDk_fun = jacfwd(jacrev(self.k_fun, argnums=0), argnums=1)
        if DDDk_fun is None:
            self.DDDk_fun = jacfwd(jacfwd(jacrev(self.k_fun, argnums=0), argnums=1), argnums=0)
        if DDDDk_fun is None:
            self.DDDDk_fun = jacfwd(jacfwd(jacfwd(jacrev(self.k_fun, argnums=0), argnums=1), argnums=0), argnums=1)
        
        FinslerManifold.__init__(self, F=self.metric, f=self.f_stanadard)
        
        return
    
    def __str__(self)->str:
        
        return "Gaussian Process Manifold with Expected Length"
    
    def f_standard(self,
                   z:Array,
                   )->Array:
        
        return self.post_mom(z)[0]
    
    def metric(self,
               z:Array,
               )->Array:
        
        mu_post, cov_post = self.jac_mom(z)
        
        if self.N_obs == 1:    
            mu_post = mu_post.reshape(1,-1)
            
        G = mu_post.dot(mu_post.T)
        
        inner_g = jnp.einsum('i,ij,j->', v, G, v)
        inner_cov = jnp.einsum('i,ij,j->', v, cov_post, v)
        omega = inner_g/inner_cov
        
        gamma1 = jscipy.special.gamma(0.5*self.emb_dim+0.5)
        gamma2 = jscipy.special.gamma(0.5*self.emb_dim)
        hyp = jscipy.special.hyp1f1(-0.5*omega, -0.5, 0.5*self.emb_dim)
        
        return jnp.sqrt(2*inner_cov)*(gamma1/gamma2)*hyp