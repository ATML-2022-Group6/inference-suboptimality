from dataclasses import dataclass
from jax import nn, random, jit
from jax import numpy as jnp
from jax.scipy import stats
from jax.example_libraries import stax

def build_flow(hps):
  n_flows: int = 2
  hidden_size: int = 50
  latent_size: int = hps.latent_size
  
  # With notation as in Dinh et al. (Real NVP paper),
  # `latent_split_size` = D//2 where the latent variable z is 
  # indexed as 1:d, d+1:D.
  latent_split_size: int = latent_size // 2
  
  def sample(rng, mu, logvar, k):
    """The forward function essentially."""
    batch_size = mu.shape[0] # B
    
    eps = random.normal(rng, (k, batch_size, latent_size)) # shape: [P, B, Z]
    z = mu + eps * jnp.exp(0.5 * logvar) # shape: [P, B, Z]
    logqz = stats.norm.logpdf(z, loc=mu, scale=logvar) # shape: [P, B]
    
    z = jnp.reshape(z, newshape=(-1, latent_size)) # shape: [P*B, Z]
    
    # With notation as in Dinh et al. (Real NVP paper),
    # split z with index 1:d, d+1:D split.
    z1, z2 = z[:, :latent_split_size], z[:, latent_split_size:]
    
    logdetsum = 0.
    # TODO :- For loop on the number of flows (has to have as many 
    # neural nets). For this implementation may need to bring current net inits
    # `_norm_flow()` to parent function init.
    z1, z2, logdet = _norm_flow(rng, z1, z2)
    logdetsum += logdet
    logdetsum = jnp.reshape(logdetsum, newshape=(k, batch_size))
    
    # Concatenate the 1:d, d+1:D z-split (will need for aux flow!).
    z = jnp.concatenate([z1, z2], axis=1)
    z = jnp.reshape(z, newshape=(k, batch_size, latent_size))
    
    logpz = logqz - logdetsum
    return logpz
  
  def _norm_flow(rng, z, v):
    """Normalizing flow implementation without Auxiliary Variable.
    Based on equation (9) and (10) in our paper Cremer et al."""
    h1_net_init, h1_net_apply = stax.serial(stax.Dense(hidden_size), stax.Tanh)
    h2_net_init, h2_net_apply = stax.serial(stax.Dense(hidden_size), stax.Tanh)
    μ1_net_init, μ1_net_apply = stax.Dense(latent_split_size)
    μ2_net_init, μ2_net_apply = stax.Dense(latent_split_size)
    σ1_net_init, σ1_net_apply = stax.Dense(latent_split_size)
    σ2_net_init, σ2_net_apply = stax.Dense(latent_split_size)

    # TODO :- Figure out these if input shapes make sense.
    _, h1_net_params = h1_net_init(rng, input_shape=(hidden_size, latent_split_size))
    _, μ1_net_params = μ1_net_init(rng, input_shape=(latent_split_size, hidden_size))
    _, σ1_net_params = σ1_net_init(rng, input_shape=(latent_split_size, hidden_size))
    _, h2_net_params = h2_net_init(rng, input_shape=(hidden_size, latent_split_size))
    _, μ2_net_params = μ2_net_init(rng, input_shape=(latent_split_size, hidden_size))
    _, σ2_net_params = σ2_net_init(rng, input_shape=(latent_split_size, hidden_size))
    
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

    logdet_v = jnp.sum(logit_1 - nn.softplus(logit_1), axis=1)
    logdet_z = jnp.sum(logit_2 - nn.softplus(logit_2), axis=1)
    logdet = logdet_v + logdet_z

    return z, v, logdet
    
  return sample