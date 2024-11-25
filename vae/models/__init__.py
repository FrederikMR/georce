#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:41:06 2024

@author: fmry
"""

#%% Modules

from .mnist import Encoder as mnist_encoder
from .mnist import Decoder as mnist_decoder
from .mnist import VAE as mnist_vae

from .svhn import Encoder as svhn_encoder
from .svhn import Decoder as svhn_decoder
from .svhn import VAE as svhn_vae

from .celeba import Encoder as celeba_encoder
from .celeba import Decoder as celeba_decoder
from .celeba import VAE as celeba_vae