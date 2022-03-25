from jax import numpy as jnp
from jax import random, grad, lax

from utils import HyperParams, log_bernoulli, log_normal
from hmc import hmc_sample_and_tune
from vae import build_vae

_, _, _, _, _, decoder = build_vae(HyperParams())

def ais_trajectory(
  rng,
  hps: HyperParams,
  decoder_params,
  x,  # Input image.
  schedule=jnp.linspace(0, 1, 500),
):
  """Annealed importance sampling trajectories for a single batch."""
  latent_size = hps.latent_size
  
  v_rng, z_rng, hmc_rng = random.split(rng, num=3)

  def _intermediate_dist(z, x, beta, log_likelihood_fn=log_bernoulli):
      zeros = jnp.zeros(z.shape)
      log_prior = log_normal(z, zeros, zeros)
      
      logit = decoder(decoder_params, z)
      log_likelihood = log_likelihood_fn(logit, x)

      return log_prior + (beta*log_likelihood)
    
  stepsize = jnp.ones(1)*0.01
  accept_trace = jnp.zeros(1)
  log_importance_weight = jnp.zeros(1)  # XC has volatile=True in torch tensor
  
  # No need backward implementation, since not doing BDMC!
  current_z = random.normal(z_rng, shape=(latent_size,))

  log_f_i = _intermediate_dist

  # WARNING :- enumerate has to start with 1 due to division by period_elapsed in HMC.
  for period_elapsed, (beta_0, beta_1) in enumerate(zip(schedule[:-1], schedule[1:]), 1):  
    period_elapsed: int

    # Log importance weight update
    log_int_1 = log_f_i(current_z, x, beta_0)
    log_int_2 = log_f_i(current_z, x, beta_1)
    log_importance_weight = log_importance_weight + (log_int_2 - log_int_1)

    def U(z):
        return -log_f_i(z, x, beta_1)

    def grad_U(z):
      gradient = grad(U)(z)
      gradient = lax.clamp(-10000., gradient, 10000.) 
      return gradient

    def normalized_K(v):
        zeros = jnp.zeros(v.shape)
        return -log_normal(v, zeros, zeros)

    tuning_params = (stepsize, accept_trace, period_elapsed)
    current_v = random.normal(v_rng, shape=current_z.shape)
    current_z, stepsize, accept_trace = hmc_sample_and_tune(
        hmc_rng, 
        current_z, current_v,
        U, normalized_K, grad_U,
        tuning_params
    )