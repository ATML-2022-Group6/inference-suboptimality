from dataclasses import dataclass
from jax import nn, random, jit
from jax import numpy as jnp
from jax.scipy import stats
from jax.example_libraries import stax

from utils import log_normal

def build_flow(hps):
  n_flows: int = 1
  hidden_size: int = 50
  latent_size: int = hps.latent_size
  
  # With notation as in Dinh et al. (Real NVP paper),
  # `latent_split_size` = D//2 where the latent variable z is 
  # indexed as 1:d, d+1:D.
  latent_split_size: int = latent_size // 2

  #              z
  #          z_1    z_2
  #      h_1         h_2
  #     /   \       /   \
  #    u1   o1     u2    o2
  #       

  h1_net_init, h1_net_apply = stax.serial(stax.Dense(hidden_size), stax.Tanh)
  h2_net_init, h2_net_apply = stax.serial(stax.Dense(hidden_size), stax.Tanh)
  μ1_net_init, μ1_net_apply = stax.Dense(latent_split_size)
  μ2_net_init, μ2_net_apply = stax.Dense(latent_split_size)
  σ1_net_init, σ1_net_apply = stax.Dense(latent_split_size)
  σ2_net_init, σ2_net_apply = stax.Dense(latent_split_size)
  
  def init_fun(rng):
    rngs = random.split(rng, num=6)

    h1_size, h1_net_params = h1_net_init(rngs[0], input_shape=(latent_split_size,))
    μ1_size, μ1_net_params = μ1_net_init(rngs[1], input_shape=h1_size)
    σ1_size, σ1_net_params = σ1_net_init(rngs[2], input_shape=h1_size)

    h2_size, h2_net_params = h2_net_init(rngs[3], input_shape=(latent_split_size,))
    μ2_size, μ2_net_params = μ2_net_init(rngs[4], input_shape=h2_size)
    σ2_size, σ2_net_params = σ2_net_init(rngs[5], input_shape=h2_size)

    params = (
      h1_net_params, μ1_net_params, σ1_net_params,
      h2_net_params, μ2_net_params, σ2_net_params
    )

    return params
  
  def sample(z0, params):
    """The forward function essentially."""
    
    # With notation as in Dinh et al. (Real NVP paper),
    # split z with index 1:d, d+1:D split.
    z1, z2 = z0[:latent_split_size], z0[latent_split_size:]
    
    logdetsum = 0.

    # TODO :- For loop on the number of flows (has to have as many 
    # neural nets). For this implementation may need to bring current net inits
    # `_norm_flow()` to parent function init.
    z1, z2, logdet = _norm_flow(params, z1, z2)
    logdetsum += logdet
    
    # Concatenate the 1:d, d+1:D z-split (will need for aux flow!).
    z = jnp.concatenate([z1, z2], axis=0)
    
    return z, logdetsum
  
  def _norm_flow(params, z, v):
    """Normalizing flow implementation without Auxiliary Variable.
    Based on equation (9) and (10) in our paper Cremer et al."""

    (
      h1_net_params, μ1_net_params, σ1_net_params,
      h2_net_params, μ2_net_params, σ2_net_params
    ) = params
    
    # Equation (9) in paper.
    h1 = h1_net_apply(h1_net_params, z)
    μ1 = μ1_net_apply(μ1_net_params, h1)
    logit_1 = σ1_net_apply(σ1_net_params, h1)
    v = v*nn.sigmoid(logit_1) + μ1  # TODO :- Add nn.elu to μ1 and multiply with `grad_fn(z)`.

    # Equation (10) in paper.
    h2 = h2_net_apply(h2_net_params, v)
    μ2 = μ2_net_apply(μ2_net_params, h2)
    logit_2 = σ2_net_apply(σ2_net_params, h2)
    z = z*nn.sigmoid(logit_2) + μ2  # TODO :- Add nn.elu to μ2 and multiply with `grad_fn(z)`.

    logdet_v = jnp.sum(logit_1 - nn.softplus(logit_1))
    logdet_z = jnp.sum(logit_2 - nn.softplus(logit_2))
    logdet = logdet_v + logdet_z

    return z, v, logdet
    
  return init_fun, sample