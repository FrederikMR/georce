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
  log_var_zx: Array

#%% Encoder

class Encoder(hk.Module):
    def __init__(self,
                 latent_dim:int=32,
                 init:hk.initializers=hk.initializers.RandomNormal(),
                 ):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.init = init
    
        self.enc1 = hk.Conv2D(output_channels=32, kernel_shape=4, stride=2, padding="SAME",
                              with_bias=False)#, w_init=self.init)
        self.enc2 = hk.Conv2D(output_channels=32, kernel_shape=4, stride=2, padding="SAME",
                              with_bias=False)#, w_init=self.init)
        self.enc3 = hk.Conv2D(output_channels=64, kernel_shape=4, stride=2, padding="SAME",
                              with_bias=False)#, w_init=self.init)
        self.enc4 = hk.Conv2D(output_channels=64, kernel_shape=4, stride=2, padding="SAME",
                              with_bias=False)#, w_init=self.init)
        # fully connected layers for learning representations
        self.fc1 = hk.Linear(output_size=256)#, w_init=self.init, b_init=self.init)
        
        self.fc_mu = hk.Linear(output_size=self.latent_dim)#, w_init=self.init, b_init=self.init)
        self.fc_log_var = hk.Linear(output_size=self.latent_dim)#, w_init=self.init, b_init=self.init)
    
    def encoder_model(self, x:Array)->Array:
        
        x = tanh(self.enc1(x))
        x = tanh(self.enc2(x))
        x = tanh(self.enc3(x))
        x = tanh(self.enc4(x))
        
        return tanh(self.fc1(x.reshape(x.shape[0], -1)))
    
    def mu_model(self, x)->Array:
        
        return self.fc_mu(x)
    
    def log_t_model(self, x)->Array:
        
        return self.fc_log_var(x)

    def __call__(self, x:Array) -> Tuple[Array, Array]:

        x_encoded = self.encoder_model(x)

        mu_zx = self.mu_model(x_encoded)
        log_var_zx = self.log_t_model(x_encoded)

        return mu_zx, log_var_zx
    
#%% Decoder

class Decoder(hk.Module):
    def __init__(self,
                 init:hk.initializers=hk.initializers.RandomNormal()):
        super(Decoder, self).__init__()
        """Decoder model."""
        self.init = init
  
        # decoder 
        self.dec1 = hk.Conv2DTranspose(output_channels=64, kernel_shape=6, stride=1, padding="SAME",
                                       with_bias=False)#, w_init=self.init)
        self.dec2 = hk.Conv2DTranspose(output_channels=64, kernel_shape=4, stride=2, padding="SAME",
                                       with_bias=False)#, w_init=self.init)
        self.dec3 = hk.Conv2DTranspose(output_channels=32, kernel_shape=4, stride=2, padding="SAME",
                                       with_bias=False)#, w_init=self.init)
        self.dec4 = hk.Conv2DTranspose(output_channels=32, kernel_shape=4, stride=2, padding="SAME",
                                       with_bias=False)#, w_init=self.init)
        self.dec5 = hk.Conv2DTranspose(output_channels=3, kernel_shape=3, stride=1, padding="SAME",
                                       with_bias=False)#, w_init=self.init)
        
        self.fc1 = hk.Linear(output_size=64*64*3)#, w_init=self.init, b_init=self.init)
  
    def decoder_model(self, x:Array)->Array:
        
        x = tanh(self.dec1(x))
        x = tanh(self.dec2(x))
        x = tanh(self.dec3(x))
        x = tanh(self.dec4(x))
        x = tanh(self.dec5(x))
        
        return self.fc1(x.reshape(x.shape[0],-1)).reshape(x.shape[0],64,64,3)
        
        #x = tanh(self.bnorm5(self.dec5(x), True))
        
        #return self.fc1(x.reshape(x.shape[0], -1)).reshape(x.shape[0], 64, 64, 3)
    
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
    
    def sample(self, mu:Array, log_var:Array):
        
        std = jnp.exp(0.5*log_var)
        eps = jrandom.normal(hk.next_rng_key(), mu.shape)
        
        return mu+std*eps

    def __call__(self, x: Array) -> VAEOutput:
        """Forward pass of the variational autoencoder."""
        mu_zx, log_var_zx = self.encoder(x)

        z = self.sample(mu_zx, log_var_zx)
        z = z.reshape(z.shape[0], 1, 1, -1)
        
        mu_xz = self.decoder(z)

        return VAEOutput(z, mu_xz, mu_zx, log_var_zx)
