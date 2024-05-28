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

#%% Load MNIST Data

def svhn_generator(data_dir:str='../../../../../Data/SVHN/',
                   seed:int=2712,
                   train_frac:float=0.8,
                   ):
    
    train_images = np.transpose(sio.loadmat(''.join((data_dir, 'train_32x32.mat')))['X'],
                                axes=(3,0,1,2))/255.
    test_images = np.transpose(sio.loadmat(''.join((data_dir, 'test_32x32.mat')))['X'],
                               axes=(3,0,1,2))/255.
    
    idx = random.choices(range(len(train_images)), k=int(len(train_images)*train_frac))
    
    ds_train = tf.data.Dataset.from_tensor_slices(train_images[idx])\
        .shuffle(buffer_size = len(train_images), seed=seed, reshuffle_each_iteration=True)
        
    ds_test = tf.data.Dataset.from_tensor_slices(test_images)\
        .shuffle(buffer_size = len(train_images), seed=seed, reshuffle_each_iteration=True)
    
    return ds_train, ds_test