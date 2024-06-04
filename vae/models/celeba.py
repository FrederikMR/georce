#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:05:11 2023

@author: fmry
"""

#%% Sources

#%% Modules

#jax,
from setup import *

#%% VAE Output

class VAEOutput(NamedTuple):
  z:Array
  mu_xz: Array
  mu_zx: Array
  std_zx: Array

#%% Encoder

class Encoder(hk.Module):
    def __init__(self,
                 latent_dim:int=32,
                 init:hk.initializers=hk.initializers.VarianceScaling(scale=2.0, 
                                                                      mode="fan_in",
                                                                      distribution="uniform"
                                                                      ),
                 ):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.init = init
    
        self.enc1 = hk.Conv2D(output_channels=32, kernel_shape=4, stride=2, padding="SAME",
                              with_bias=False, w_init=self.init)
        self.enc2 = hk.Conv2D(output_channels=32, kernel_shape=4, stride=2, padding="SAME",
                              with_bias=False, w_init=self.init)
        self.enc3 = hk.Conv2D(output_channels=64, kernel_shape=4, stride=2, padding="SAME",
                              with_bias=False, w_init=self.init)
        self.enc4 = hk.Conv2D(output_channels=64, kernel_shape=4, stride=2, padding="SAME",
                              with_bias=False, w_init=self.init)
        # fully connected layers for learning representations
        self.fc1 = hk.Linear(output_size=256, w_init=self.init, b_init=self.init)
        
        self.fc_mu = hk.Linear(output_size=self.latent_dim, w_init=self.init, b_init=self.init)
        self.fc_std = hk.Linear(output_size=self.latent_dim, w_init=self.init, b_init=self.init)
    
    def encoder_model(self, x:Array)->Array:
        
        x = gelu(self.enc1(x))
        x = gelu(self.enc2(x))
        x = gelu(self.enc3(x))
        x = gelu(self.enc4(x))
        
        return gelu(self.fc1(x.reshape(x.shape[0], -1)))
    
    def mu_model(self, x)->Array:
        
        return self.fc_mu(x)
    
    def std_model(self, x)->Array:
        
        return sigmoid(self.fc_std(x))

    def __call__(self, x:Array) -> Tuple[Array, Array]:

        x_encoded = self.encoder_model(x)

        mu_zx = self.mu_model(x_encoded)
        std_zx = self.std_model(x_encoded)

        return mu_zx, std_zx
    
#%% Decoder

class Decoder(hk.Module):
    def __init__(self,
                 init:hk.initializers=hk.initializers.VarianceScaling(scale=2.0, 
                                                                      mode="fan_in",
                                                                      distribution="uniform",
                                                                      ),
                 ):
        super(Decoder, self).__init__()
        """Decoder model."""
        self.init = init
  
        # decoder 
        self.dec1 = hk.Conv2DTranspose(output_channels=64, kernel_shape=4, stride=2, padding="SAME",
                                       with_bias=False, w_init=self.init)
        self.dec2 = hk.Conv2DTranspose(output_channels=64, kernel_shape=4, stride=2, padding="SAME",
                                       with_bias=False, w_init=self.init)
        self.dec3 = hk.Conv2DTranspose(output_channels=32, kernel_shape=4, stride=2, padding="SAME",
                                       with_bias=False, w_init=self.init)
        self.dec4 = hk.Conv2DTranspose(output_channels=32, kernel_shape=4, stride=2, padding="SAME",
                                       with_bias=False, w_init=self.init)
        self.dec5 = hk.Conv2DTranspose(output_channels=3, kernel_shape=4, stride=4, padding="SAME",
                                       with_bias=False, w_init=self.init)
  
    def decoder_model(self, x:Array)->Array:
        
        x = gelu(self.dec1(x))
        x = gelu(self.dec2(x))
        x = gelu(self.dec3(x))
        x = gelu(self.dec4(x))
        x = self.dec5(x)
        
        return x
    
    def __call__(self, z: Array) -> Array:
        
        return self.decoder_model(z)
  
#%% Riemannian Score Variational Prior

class VAE(hk.Module):
    def __init__(self,
                 encoder:Encoder,
                 decoder:Decoder,
                 seed:int=2712,
                 ):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.key = jrandom.key(seed)
    
    def sample(self, mu:Array, std:Array):
        
        return mu+std*jrandom.normal(hk.next_rng_key(), mu.shape)

    def __call__(self, x: Array) -> VAEOutput:
        """Forward pass of the variational autoencoder."""
        mu_zx, std_zx = self.encoder(x)

        z = self.sample(mu_zx, std_zx)
        z = z.reshape(z.shape[0], 1, 1, -1)
        
        mu_xz = self.decoder(z)

        return VAEOutput(z, mu_xz, mu_zx, std_zx)
