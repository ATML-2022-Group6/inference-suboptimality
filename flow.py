from jax import nn
from jax import random
from jax import numpy as jnp
from jax.example_libraries import stax

def norm_flow():
    rng = random.PRNGKey(0)
    
    z_size: int
    z: any
    v: any
    h_s: int = 50
    
    h1_net_init, h1_net_apply = stax.serial(stax.Dense(h_s), stax.Tanh)
    h2_net_init, h2_net_apply = stax.serial(stax.Dense(h_s), stax.Tanh)
    μ1_net_init, μ1_net_apply = stax.Dense(z_size)
    μ2_net_init, μ2_net_apply = stax.Dense(z_size)
    σ1_net_init, σ1_net_apply = stax.Dense(z_size)
    σ2_net_init, σ2_net_apply = stax.Dense(z_size)
    
    _, h1_net_params = h1_net_init(rng, input_shape=(z_size, h_s))
    _, μ1_net_params = μ1_net_init(rng, input_shape=(h_s, z_size))
    _, σ1_net_params = σ1_net_init(rng, input_shape=(h_s, z_size))
    _, h2_net_params = h2_net_init(rng, input_shape=(z_size, h_s))
    _, μ2_net_params = μ2_net_init(rng, input_shape=(h_s, z_size))
    _, σ2_net_params = σ2_net_init(rng, input_shape=(h_s, z_size))
    
    h1 = h1_net_apply(h1_net_params, z)
    μ1 = μ1_net_apply(μ1_net_params, h1)
    logit_1 = σ1_net_apply(σ1_net_params, h1)
    v = v*nn.sigmoid(logit_1) + μ1
    
    h2 = h2_net_apply(h2_net_params, v)
    μ2 = μ2_net_apply(μ2_net_params, h2)
    logit_2 = σ2_net_apply(σ2_net_params, h2)
    z = z*nn.sigmoid(logit_2) + μ2
    
    logdet_v = jnp.sum(logit_1 - nn.softplus(logit_1), axis=1)
    logdet_z = jnp.sum(logit_2 - nn.softplus(logit_2), axis=1)
    logdet = logdet_v + logdet_z
    
    return z, v, logdet