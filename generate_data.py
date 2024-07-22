#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:11:07 2023

@author: fmry
"""

#%% Sources

#%% Modules

#jax
import jax.numpy as jnp

#scipy
from scipy import ndimage

#os
import os

#regression
from geometry.regression import GPRegression

#vae
from vae.model_loader import mnist_generator

#%% train for (x,y,t)

def generate_data(manifold:str="gp_mnist")->None:
    
    if manifold == 'gp_mnist':
        
        num_rotate = 200
        
        mnist_dataloader = mnist_generator(seed=2712,
                                           batch_size=64,
                                           split='train[:80%]')
        mnist_data = next(mnist_dataloader).x[2]
        
        theta = jnp.linspace(0,2*jnp.pi,num_rotate)
        x1 = jnp.cos(theta)
        x2 = jnp.sin(theta)
        
        theta_degrees = theta*180/jnp.pi
        
        rot = []
        for v in theta_degrees:
            rot.append(ndimage.rotate(mnist_data, v, reshape=False))
        rot = jnp.stack(rot)/255
        
        if not os.path.exists('data/MNIST/'):
            os.makedirs('data/MNIST/')
        
        jnp.save('data/MNIST/y.npy', rot.reshape(rot.shape[0],-1).T)
        jnp.save('data/MNIST/X.npy', jnp.vstack((x1,x2)))
        
        return
    
    return

#%% Main

if __name__ == '__main__':
        
    generate_data()
    
