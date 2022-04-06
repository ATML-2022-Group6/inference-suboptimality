import jax
from jax import numpy as jnp
from jax import random, grad, lax, jit
from jax.scipy.special import logsumexp
from functools import partial
from utils import log_bernoulli, log_normal
from hmc import hmc_sample_and_tune
from vae import VAE

from dataclasses import dataclass

@dataclass(eq=True, frozen=True)
class AISHyperParams:
  num_iwae_samples: int = 10
  annealing_steps: int = 10000

@partial(jit, static_argnums=(0,1))
def ais_trajectory(hps: AISHyperParams, model: VAE, decoder_params, image, rng):
  """Annealed importance sampling trajectories for a single batch."""
  latent_size = model.hps.latent_size

  v_rng, z_rng, hmc_rng = random.split(rng, num=3)

  def log_f_i(z, beta):
    log_prior = log_normal(z)
    logit = model.decoder(decoder_params, z)
    log_likelihood = log_bernoulli(logit, image)

    return log_prior + beta * log_likelihood

  def ais_step(accum, args):
    current_z, stepsize, accept_trace = accum
    beta_0, beta_1, period_elapsed = args

    log_int_1 = log_f_i(current_z, beta_0)
    log_int_2 = log_f_i(current_z, beta_1)
    log_importance_weight = log_int_2 - log_int_1

    def U(z):
      return -log_f_i(z, beta_1)

    def grad_U(z):
      gradient = grad(U)(z)
      gradient = lax.clamp(-10000., gradient, 10000.)
      return gradient

    def normalized_K(v):
      return -log_normal(v)

    tuning_params = (stepsize, accept_trace, period_elapsed)
    current_v = random.normal(random.fold_in(v_rng, period_elapsed), shape=current_z.shape)
    new_accum = hmc_sample_and_tune(
      random.fold_in(hmc_rng, period_elapsed),
      current_z, current_v,
      U, normalized_K, grad_U,
      tuning_params
    )

    return new_accum, log_importance_weight

  stepsize = 0.01
  accept_trace = 0.0
  init_z = random.normal(z_rng, shape=(latent_size,))
  init_accum = (init_z, stepsize, accept_trace)

  annealing_schedule = jnp.linspace(0., 1., hps.annealing_steps)
  scan_args = (
    annealing_schedule[:-1], annealing_schedule[1:], jnp.arange(1, hps.annealing_steps)
  )
  _, log_weights = jax.lax.scan(ais_step, init_accum, scan_args)

  return jnp.sum(log_weights)

@partial(jit, static_argnums=(0, 1))
def ais_iwelbo(hps: AISHyperParams, model, decoder_params, image, rng):
  rngs = random.split(rng, hps.num_iwae_samples)
  logws = jax.vmap(ais_trajectory, in_axes=(None, None, None, None, 0))(
    hps, model, decoder_params, image, rngs
  )
  logw_iwae = logsumexp(logws) - jnp.log(hps.num_iwae_samples)
  return logw_iwae

@partial(jit, static_argnums=(0, 1))
def batch_ais_iwelbo(hps, model, decoder_params, images, rng):
  rngs = random.split(rng, len(images))
  logw_iwaes = jax.vmap(ais_iwelbo, in_axes=(None, None, None, 0, 0))(
    hps, model, decoder_params, images, rngs
  )
  return jnp.mean(logw_iwaes)