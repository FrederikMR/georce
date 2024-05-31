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

def mnist_generator(seed:int=2712,
                    split:str='train[:80%]', 
                    batch_size:int=100,
                    )->Iterator[Batch]:
  ds = (
      tfds.load("mnist", split=split)
      .shuffle(buffer_size=10 * batch_size, seed=seed)
      .batch(batch_size)
      .prefetch(buffer_size=5)
      .repeat()
      .as_numpy_iterator()
  )

  return map(lambda x: Batch(x['image']/255.0), ds)

#def mnist_generator(seed:int=2712,
#                    train_frac:float=0.8,
#                    ):
#    
#    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
#    
#    idx = random.choices(range(len(train_images)), k=int(len(train_images)*train_frac))
#    
#    train_images = preprocess_images(train_images[idx])
#    test_images = preprocess_images(test_images)
#    
#    ds_train = tf.data.Dataset.from_tensor_slices(train_images)\
#        .shuffle(buffer_size = len(train_images), seed=seed, reshuffle_each_iteration=True)
#        
#    ds_test = tf.data.Dataset.from_tensor_slices(test_images)\
#        .shuffle(buffer_size = len(train_images), seed=seed, reshuffle_each_iteration=True)
#    
#    return ds_train, ds_test