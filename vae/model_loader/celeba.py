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

def celeba_generator(data_dir:str='../../../../../Data/CelebA/',
                     seed:int=2712,
                     train_frac:float=0.8,
                     ):
    
    if not(os.path.exists(data_dir)):
        os.mkdir(data_dir)
    
    data_dir = ''.join((data_dir, 'celeba.zip'))
    
    if not (os.path.isfile(data_dir)):
    
        url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
        gdown.download(url, data_dir, quiet=True)
    
        with ZipFile(data_dir, "r") as zipobj:
            zipobj.extractall("celeba_gan")
        
    ds_train = keras.utils.image_dataset_from_directory(
        "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=1,
        validation_split=1-train_frac, subset="training", seed=seed)
    ds_train = ds_train.map(lambda x: x / 255.0)
    
    ds_test = keras.utils.image_dataset_from_directory(
        "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=1,
        validation_split=1-train_frac, subset="validation", seed=seed)
    ds_test = ds_test.map(lambda x: x / 255.0)
    
    return ds_train, ds_test
    
    #ds_train = tf.keras.utils.image_dataset_from_directory(
    #    data_dir,
    #    validation_split=0.2,
    #    subset="training",
    #    seed=seed,
    #    image_size=(64, 64))
    #
    #ds_test = tf.keras.utils.image_dataset_from_directory(
    #      data_dir,
    #      validation_split=0.2,
    #      subset="validation",
    #      seed=seed,
    #      image_size=(64, 64))
    
    #ds_train = ds_train.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y))
    #ds_test = ds_train.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y))
    
    #return ds_train, ds_test