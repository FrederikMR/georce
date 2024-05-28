#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:47:00 2024

@author: fmry
"""

#%% Modules

from jax import Array
from jax import tree_leaves, tree_map, tree_flatten, tree_unflatten
from jax import vmap, grad, jacfwd, value_and_grad, jit
from jax import lax
from jax.nn import tanh, sigmoid

import jax.numpy as jnp
import jax.random as jrandom

#JAX Optimization
from jax.example_libraries import optimizers

import numpy as np

from scipy.optimize import minimize
import scipy.io as sio

import random

import tensorflow as tf

import keras

#haiku
import haiku as hk

#optax
import optax

#pickle
import pickle

#os
import os

import gdown

from zipfile import ZipFile

from abc import ABC
from typing import Callable, Tuple, NamedTuple