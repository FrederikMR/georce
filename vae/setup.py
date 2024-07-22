#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:36:12 2024

@author: fmry
"""

#%% Modules

from jax import Array
from jax import value_and_grad, jit
from jax import tree_leaves, tree_map, tree_flatten, tree_unflatten
from jax.nn import gelu, sigmoid

import jax.numpy as jnp
import jax.random as jrandom

import numpy as np

#haiku
import haiku as hk

#optax
import optax

#tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds

#os
import os

#pickle
import pickle

#gdown
import gdown

#zipfile
from zipfile import ZipFile

from typing import Tuple, NamedTuple, Iterator, Callable