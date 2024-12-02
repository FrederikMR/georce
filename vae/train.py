#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 00:08:19 2024

@author: fmry
"""

#%% Sources


#%% Modules

from vae.setup import *

from .model_loader import save_model

#%% TrainingState

class TrainingState(NamedTuple):
    params: hk.Params
    state_val:  dict
    opt_state: optax.OptState
    rng_key: Array
    
#%% Pre-Train VAE

def train_VAE_model(model:object,
                    generator:object,
                    lr_rate:float = 0.002,
                    save_path:str = '',
                    state:TrainingState = None,
                    epochs:int=1000,
                    save_step:int = 100,
                    optimizer:object = None,
                    seed:int=2712,
                    criterion:Callable[[Array, Array],Array]=None,
                    )->None:
    
    @jit
    def loss_fun(params:hk.Params, state, x:Array)->Array:
        
        z, mu_xz, mu_zx, std_zx = vae_apply_fn(params, x, state.rng_key, state.state_val)
        
        batch = z.shape[0]

        x = x.reshape(batch, -1)
        z = z.reshape(batch, -1)
        mu_xz = mu_xz.reshape(batch, -1)
        mu_zx = mu_zx.reshape(batch, -1)
        std_zx = std_zx.reshape(batch, -1)

        rec_loss = criterion(mu_xz, x)
        
        var_zx = std_zx**2
        log_var_zx = jnp.log(var_zx)
        kld = 0.5 * jnp.sum(-log_var_zx - 1. + var_zx + jnp.square(mu_zx), axis=-1)
        elbo = kld-rec_loss
        
        return jnp.mean(elbo), (jnp.mean(rec_loss), jnp.mean(kld))
    
    @jit
    def update(state:TrainingState, data:Array):
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        
        loss, gradients = value_and_grad(loss_fun, has_aux=True)(state.params,
                                                                 state,
                                                                 data
                                                                 )
        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
    if criterion is None:
        criterion = jit(lambda x,y: -jnp.sum(jnp.square(x-y), axis=-1))
    
    if optimizer is None:
        optimizer = optax.adam(learning_rate = lr_rate,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    initial_rng_key = jrandom.PRNGKey(seed)

    data_sample = next(generator).x
    if type(model) == hk.Transformed:
        if state is None:
            initial_params = model.init(jrandom.PRNGKey(seed), data_sample)
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: model.apply(params, rng_key, data)
    elif type(model) == hk.TransformedWithState:
        if state is None:
            initial_params, init_state = model.init(jrandom.PRNGKey(seed), data_sample)
            initial_opt_state = optimizer.init(initial_params)
            state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: model.apply(params, state_val, rng_key, data)[0]

    for step in range(epochs):
        state, loss = update(state, next(generator).x)
        if (step+1) % save_step == 0:
            save_model(save_path, state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
    save_model(save_path, state)
    
    return