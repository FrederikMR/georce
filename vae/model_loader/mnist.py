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

#%% Preprocess images

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

#%% Load MNIST Data

def mnist_generator(seed:int=2712
                    ):
    
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    
    ds_train = tf.data.Dataset.from_tensor_slices(train_images)\
        .shuffle(buffer_size = len(train_images), seed=seed, reshuffle_each_iteration=True)
        
    ds_test = tf.data.Dataset.from_tensor_slices(test_images)\
        .shuffle(buffer_size = len(train_images), seed=seed, reshuffle_each_iteration=True)
    
    return ds_train, ds_test