#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:40:59 2024

@author: fmry
"""

#%% Sources

#https://www.tensorflow.org/tutorials/generative/cvae

#%% Modules

from setup import *

#%% Batch class

class Batch(NamedTuple):
    x: Array  # [B, H, W, C]x

#%% Load MNIST Data

def svhn_generator(data_dir:str="../../../Data/SVHN/",
                   split:str='train[:80%]', 
                   batch_size: int=100, 
                   seed: int=2712
                   )->Iterator[Batch]:
  ds = (
      tfds.load("svhn_cropped", split=split, data_dir=data_dir, download=True)
      .shuffle(buffer_size=10 * batch_size, seed=seed)
      .batch(batch_size)
      .prefetch(buffer_size=5)
      .repeat()
      .as_numpy_iterator()
  )
  return map(lambda x: Batch(x["image"]/255.0), ds)