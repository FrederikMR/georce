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

import os

from vae.model_loader import mnist_generator, svhn_generator, celeba_generator, load_model

from vae.models import mnist_encoder
from vae.models import mnist_decoder
from vae.models import mnist_vae

from vae.models import svhn_encoder
from vae.models import svhn_decoder
from vae.models import svhn_vae

from vae.models import celeba_encoder
from vae.models import celeba_decoder
from vae.models import celeba_vae

#%% Load manifolds

def generate_data(manifold:str="celeba", 
                  dim:int = 32,
                  data_path:str = 'data/',
                  svhn_path:str = "../../../Data/SVHN/",
                  celeba_path:str = "../../../Data/CelebA/",
                  ):

    if manifold == "celeba":
        
        save_path = ''.join((data_path, f'celeba_{dim}/'))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        celeba_state = load_model(''.join(('models/', f'celeba_{dim}/')))
        celeba_dataloader = celeba_generator(data_dir=celeba_path,
                                             batch_size=64,
                                             seed=2712,
                                             split=0.8,
                                             )
        @hk.transform
        def celeba_tvae(x):

            vae = celeba_vae(
                        encoder=celeba_encoder(latent_dim=dim),
                        decoder=celeba_decoder(),
            )
         
            return vae(x)
        
        celeba_vae_fun = jit(lambda x: celeba_tvae.apply(lax.stop_gradient(celeba_state.params),
                                                         celeba_state.rng_key,
                                                         x))
       
       
        celeba_data = next(celeba_dataloader).x
        celeba_rec = celeba_vae_fun(celeba_data)
       
        z0, zT = celeba_rec.mu_zx[0], celeba_rec.mu_zx[1]
        
        jnp.save(''.join((save_path, 'z0.npy')), z0)
        jnp.save(''.join((save_path, 'zT.npy')), zT)
        
        return
   
    if manifold == "svhn":
        
        save_path = ''.join((data_path, f'svhn_{dim}/'))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        svhn_state = load_model(''.join(('models/', f'svhn_{dim}/')))
        svhn_dataloader = svhn_generator(data_dir=svhn_path,
                                         batch_size=64,
                                         seed=2712,
                                         split='train[:80%]',
                                         )
        @hk.transform
        def svhn_tvae(x):

            vae = svhn_vae(
                        encoder=svhn_encoder(latent_dim=dim),
                        decoder=svhn_decoder(),
            )
         
            return vae(x)

        svhn_vae_fun = jit(lambda x: svhn_tvae.apply(lax.stop_gradient(svhn_state.params),
                                                     svhn_state.rng_key,
                                                     x))
       
       
        svhn_data = next(svhn_dataloader).x
        svhn_rec = svhn_vae_fun(svhn_data)
        
        z0, zT = svhn_rec.mu_zx[0], svhn_rec.mu_zx[1]
        
        jnp.save(''.join((save_path, 'z0.npy')), z0)
        jnp.save(''.join((save_path, 'zT.npy')), zT)
        
        return
        
    elif manifold == "mnist":
        
        save_path = ''.join((data_path, f'mnist_{dim}/'))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        mnist_state = load_model(''.join(('models/', f'mnist_{dim}/')))
        mnist_dataloader = mnist_generator(seed=2712,
                                           batch_size=64,
                                           split='train[:80%]')
       
        @hk.transform
        def mnist_tvae(x):
       
            vae = mnist_vae(
                        encoder=mnist_encoder(latent_dim=dim),
                        decoder=mnist_decoder(),
            )
       
            return vae(x)
        mnist_vae_fun = lambda x: mnist_tvae.apply(mnist_state.params,
                                                   mnist_state.rng_key,
                                                   x)
       
       
        mnist_data = next(mnist_dataloader).x
        mnist_rec = mnist_vae_fun(mnist_data)
       
        z0, zT = mnist_rec.mu_zx[0], mnist_rec.mu_zx[1]
        
        jnp.save(''.join((save_path, 'z0.npy')), z0)
        jnp.save(''.join((save_path, 'zT.npy')), zT)
        
        return
        
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")
        
    return