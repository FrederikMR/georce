#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:04:42 2024

@author: fmry
"""

#%% Import packages

#jax
import jax.numpy as jnp
from jax import Array, jacfwd, lax, vmap

#typing
from typing import Callable