#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:42 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp

#argparse
import argparse

#time
import time

import haiku as hk

#os
import os

#Typing
from typing import List

#Pickle
import pickle

from vae import train_VAE_model
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

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--model', default="celeba",
                        type=str)
    parser.add_argument('--svhn_dir', default="../../../Data/SVHN/",
                        type=str)
    parser.add_argument('--celeba_dir', default="../../../Data/CelebA/",
                        type=str)
    parser.add_argument('--lr_rate', default=0.0002,
                        type=float)
    parser.add_argument('--con_training', default=0,
                        type=int)
    parser.add_argument('--split', default=0.8,
                        type=float)
    parser.add_argument('--batch_size', default=100,
                        type=int)
    parser.add_argument('--epochs', default=50000,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)
    parser.add_argument('--save_step', default=100,
                        type=int)
    parser.add_argument('--save_path', default='models/',
                        type=str)

    args = parser.parse_args()
    return args

#%% Load manifolds

def train_vae()->None:
    
    args = parse_args()
    
    if args.model == "mnist":
        split = int(args.split*100)
        ds_train = mnist_generator(seed=args.seed, 
                                   batch_size=args.batch_size,
                                   split=f'train[:{split}%]')
        
        @hk.transform
        def vae_model(x):
            vae = mnist_vae(
                        encoder=mnist_encoder(latent_dim=8),
                        decoder=mnist_decoder(),
            )
          
            return vae(x)
    elif args.model == "svhn":
        split = int(args.split*100)
        ds_train = svhn_generator(data_dir=args.svhn_dir,
                                  batch_size=args.batch_size,
                                  seed=args.seed, 
                                  split=f'train[:{split}%]')
        
        @hk.transform
        def vae_model(x):

            vae = svhn_vae(
                        encoder=svhn_encoder(latent_dim=32),
                        decoder=svhn_decoder(),
            )
          
            return vae(x)
    elif args.model == "celeba":
        ds_train = celeba_generator(data_dir=args.celeba_dir,
                                    batch_size=args.batch_size,
                                    seed=args.seed, 
                                    split=args.split)
        
        @hk.transform
        def vae_model(x):

            vae = celeba_vae(
                        encoder=celeba_encoder(latent_dim=32),
                        decoder=celeba_decoder(),
            )
          
            return vae(x)
    else:
        raise ValueError(f"Undefined data model {args.model}. You can only choose: mnist, svhn, celeba")
                            
    save_path = ''.join((args.save_path, args.model, '/'))
    if args.con_training:
        state = load_model(save_path)
    else:
        state = None
    
    if not (os.path.exists(save_path)):
        os.mkdir(save_path)

    train_VAE_model(model=vae_model,
                        generator=ds_train,
                        lr_rate = args.lr_rate,
                        save_path = save_path,
                        state = state,
                        epochs=args.epochs,
                        save_step = args.save_step,
                        optimizer = None,
                        seed=args.seed,
                        criterion=None,
                        )
    
    
    return

#%% Main

if __name__ == '__main__':
    
    train_vae()