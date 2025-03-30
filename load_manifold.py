#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:42 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp
from jax import jit, lax

import haiku as hk

from geometry.manifolds.riemannian import nSphere, nEllipsoid, nEuclidean, \
    nParaboloid, HyperbolicParaboloid, SPDN, H2, Cylinder, Landmarks, T2, LatentSpaceManifold, FisherRaoGeometry
    
from vae.model_loader import mnist_generator, svhn_generator, celeba_generator, load_model

from vae.models import mnist_encoder
from vae.models import mnist_decoder

from vae.models import svhn_encoder
from vae.models import svhn_decoder

from vae.models import celeba_encoder
from vae.models import celeba_decoder

#%% Load manifolds

def load_manifold(manifold:str="Euclidean", 
                  dim:int = 2,
                  svhn_path:str = "../../../Data/SVHN/",
                  celeba_path:str = "../../../Data/CelebA/",
                  ):
    
    rho = 0.5 #default
    if manifold == "Euclidean":
        M = nEuclidean(dim=dim)
        z0 = -jnp.linspace(0,1,dim)
        zT = jnp.ones(dim, dtype=jnp.float32)
        rho = 0.5
    if manifold == "SPDN":
        M = SPDN(N=dim)
        x0 = jnp.eye(dim)
        
        z0 = M.invf(x0)
        zT = jnp.linspace(0.5,1.0, M.dim)
        rho = 0.5
    elif manifold == "Paraboloid":
        M = nParaboloid(dim=dim)
        z0 = -jnp.linspace(0,1,dim)
        zT = jnp.ones(dim, dtype=jnp.float32)
        rho = 0.5    
    elif manifold == "Sphere":
        M = nSphere(dim=dim)
        z0 = -jnp.linspace(0,1,dim)
        zT = 0.5*jnp.ones(dim, dtype=jnp.float32)
        rho = .5
    elif manifold == "Ellipsoid":
        params = jnp.linspace(0.5,1.0,dim+1)
        M = nEllipsoid(dim=dim, params=params)
        z0 = -jnp.linspace(0,1,dim)
        zT = 0.5*jnp.ones(dim, dtype=jnp.float32)
        rho = 0.5
    elif manifold == "H2":
        M = H2()
        z0 = jnp.array([1.0,1.0])
        zT = jnp.array([0.1, 0.1])
        rho = 0.5
    elif manifold == "Cylinder":
        M = Cylinder()
        z0 = jnp.array([-5*jnp.pi/4,1.0])
        zT = jnp.array([jnp.pi, -1.0])
        rho = 0.5
    elif manifold == "T2":
        M = T2(R=3.0, r=1.0)
        z0 = jnp.array([0.0, 0.0])
        zT = jnp.array([5*jnp.pi/4, 5*jnp.pi/4])
        rho = 0.5
    elif manifold == "Landmarks":
        M = Landmarks(N=dim, m=2, k_alpha=0.1)
        z0 = jnp.vstack((jnp.linspace(-5.0,5.0,M.N),jnp.linspace(0.0,0.0,M.N))).T.flatten()
        zT = jnp.vstack((jnp.linspace(-1.0,1.0,M.N),jnp.linspace(0.0, 0.0, M.N))).T.flatten()
        rho = 0.5
    elif manifold == "Gaussian":
        M = FisherRaoGeometry(distribution='Gaussian')
        z0 = jnp.array([-1.0, 0.5])
        zT = jnp.array([1.0, 1.0])
        rho = .5
    elif manifold == "Frechet":
        M = FisherRaoGeometry(distribution='Frechet')
        z0 = jnp.array([0.5, 0.5])
        zT = jnp.array([1.0, 1.0])
        rho = 0.5
    elif manifold == "Cauchy":
        M = FisherRaoGeometry(distribution='Cauchy')
        z0 = jnp.array([-1.0, 0.5])
        zT = jnp.array([1.0, 1.0])
        rho = 0.5
    elif manifold == "Pareto":
        M = FisherRaoGeometry(distribution='Pareto')
        z0 = jnp.array([0.5, 0.5])
        zT = jnp.array([1.0, 1.0])
        rho = 0.5
    elif manifold == "celeba":
        celeba_state = load_model(''.join(('models/', f'celeba_{dim}/')))
       
        @hk.transform
        def celeba_tencoder(x):
       
            encoder = celeba_encoder(latent_dim=32)
       
            return encoder(x)[0]
       
        @hk.transform
        def celeba_tdecoder(x):
       
            decoder = celeba_decoder()
       
            return decoder(x)
       
        celeba_encoder_fun = jit(lambda x: celeba_tencoder.apply(lax.stop_gradient(celeba_state.params),
                                                                 None,
                                                                 x.reshape(-1,64,64,3)
                                                                 )[0].reshape(-1,dim).squeeze())
        celeba_decoder_fun = jit(lambda x: celeba_tdecoder.apply(lax.stop_gradient(celeba_state.params),
                                                                 None,
                                                                 x.reshape(-1,dim)
                                                                 ).reshape(-1,64*64*3).squeeze())
       
        M = LatentSpaceManifold(dim=dim,
                                emb_dim=64*64*3,
                                encoder=celeba_encoder_fun,
                                decoder=celeba_decoder_fun,
                                )
        
        z0 = jnp.load(f'data/celeba_{dim}/z0.npy')
        zT = jnp.load(f'data/celeba_{dim}/zT.npy')
       
        return z0, zT, M, rho
   
    elif manifold == "svhn":
        svhn_state = load_model(''.join(('models/', f'svhn_{dim}/')))
       
        @hk.transform
        def svhn_tencoder(x):
       
            encoder = svhn_encoder(latent_dim=dim)
       
            return encoder(x)[0]
       
        @hk.transform
        def svhn_tdecoder(x):
       
            decoder = svhn_decoder()
       
            return decoder(x)
       
        svhn_encoder_fun = jit(lambda x: svhn_tencoder.apply(lax.stop_gradient(svhn_state.params),
                                                             None,
                                                             x.reshape(-1,32,32,3)
                                                             )[0].reshape(-1,dim).squeeze())
        svhn_decoder_fun = jit(lambda x: svhn_tdecoder.apply(lax.stop_gradient(svhn_state.params),
                                                             None,
                                                             x.reshape(-1,dim)
                                                             ).reshape(-1,32*32*3).squeeze())
       
        M = LatentSpaceManifold(dim=dim,
                                emb_dim=32*32*3,
                                encoder=svhn_encoder_fun,
                                decoder=svhn_decoder_fun,
                                )
        
        z0 = jnp.load(f'data/svhn_{dim}/z0.npy')
        zT = jnp.load(f'data/svhn_{dim}/zT.npy')

        return z0, zT, M, rho
    elif manifold == "mnist":
        mnist_state = load_model(''.join(('models/', f'mnist_{dim}/')))
       
        @hk.transform
        def mnist_tencoder(x):
       
            encoder = mnist_encoder(latent_dim=dim)
       
            return encoder(x)[0]
       
        @hk.transform
        def mnist_tdecoder(x):
       
            decoder = mnist_decoder()
       
            return decoder(x)
       
        mnist_encoder_fun = lambda x: mnist_tencoder.apply(mnist_state.params,
                                                           None,
                                                           x.reshape(-1,28,28,1)
                                                           )[0].reshape(-1,dim).squeeze()
        mnist_decoder_fun = lambda x: mnist_tdecoder.apply(mnist_state.params,
                                                           None,
                                                           x.reshape(-1,dim)
                                                           ).reshape(-1,28*28).squeeze()
       
        M = LatentSpaceManifold(dim=dim,
                                emb_dim=28*28,
                                encoder=mnist_encoder_fun,
                                decoder=mnist_decoder_fun,
                                )
        
        z0 = jnp.load(f'data/mnist_{dim}/z0.npy')
        zT = jnp.load(f'data/mnist_{dim}/zT.npy')
       
        return z0, zT, M, rho
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")
        
    return z0, zT, M, rho