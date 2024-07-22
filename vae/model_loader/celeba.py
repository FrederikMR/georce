#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:40:59 2024

@author: fmry
"""

#%% Sources

#https://www.tensorflow.org/tutorials/generative/cvae

#%% Modules

from vae.setup import *

#%% Batch class

class Batch(NamedTuple):
    x: Array  # [B, H, W, C]x

#%% Load MNIST Data

def celeba_generator(data_dir:str='../../../../../Data/CelebA/',
                     img_size:Tuple[int, int] = (64, 64), 
                     batch_size: int=100, 
                     split:float=.8,
                     seed: int=2712
                     )->Iterator[Batch]:
    
    def preprocess_image(filename:str):
        
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, img_size)
        
        return image
    
    if not(os.path.exists(data_dir)):
        os.mkdir(data_dir)
        
    zip_dir = ''.join((data_dir, 'img'))
    data_dir = ''.join((data_dir, 'celeba.zip'))
    
    if not (os.path.isfile(data_dir)):
    
        url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
        gdown.download(url, data_dir, quiet=True)
    
    if not(os.path.exists(zip_dir)):
        os.mkdir(zip_dir)
        with ZipFile(data_dir, "r") as zipobj:
            zipobj.extractall(zip_dir)

    img_dir = ''.join((zip_dir, '/img_align_celeba/'))
    filenames = tf.constant([os.path.join(img_dir, fname) for fname in os.listdir(img_dir)])
    dataset = tf.data.Dataset.from_tensor_slices((filenames[:int(len(filenames)*split)]))

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.repeat().as_numpy_iterator()
    
    return map(lambda x: Batch(x), dataset)

#def celeba_generator(data_dir:str='../../../../../Data/CelebA/',
#                     seed:int=2712,
#                     train_frac:float=0.8,
#                     ):
#    
#    if not(os.path.exists(data_dir)):
#        os.mkdir(data_dir)
#        
#    zip_dir = ''.join((data_dir, 'img'))
#    data_dir = ''.join((data_dir, 'celeba.zip'))
#    
#    if not (os.path.isfile(data_dir)):
#    
#        url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
#        gdown.download(url, data_dir, quiet=True)
#    
#    if not(os.path.exists(zip_dir)):
#        os.mkdir(zip_dir)
#        with ZipFile(data_dir, "r") as zipobj:
#            zipobj.extractall(zip_dir)
#        
#    ds_train = keras.utils.image_dataset_from_directory(
#        zip_dir, label_mode=None, image_size=(64, 64), batch_size=1,
#        validation_split=1-train_frac, subset="training", seed=seed)
#    ds_train = ds_train.map(lambda x: x / 255.0)
#    
#    ds_test = keras.utils.image_dataset_from_directory(
#        zip_dir, label_mode=None, image_size=(64, 64), batch_size=1,
#        validation_split=1-train_frac, subset="validation", seed=seed)
#    ds_test = ds_test.map(lambda x: x / 255.0)
#    
#    return ds_train, ds_test
    
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