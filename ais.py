import jax
from jax import numpy as jnp
from jax import random, grad, lax, jit
from jax.scipy.special import logsumexp
from functools import partial
from utils import log_bernoulli, log_normal
from hmc import hmc_sample_and_tune


@partial(jit, static_argnums=(1,))
def ais_trajectory(
  rng,
  model,
  decoder_params,
  x,  # Input image.
  annealing_schedule=jnp.linspace(0., 1., 10000),
):
  """Annealed importance sampling trajectories for a single batch."""
  latent_size = model.hps.latent_size

  v_rng, z_rng, hmc_rng = random.split(rng, num=3)

  def log_f_i(z, x, beta):
    log_prior = log_normal(z)
    logit = model.decoder(decoder_params, z)
    log_likelihood = log_bernoulli(logit, x)

    return log_prior + beta * log_likelihood

  def ais_step(accum, args):
    current_z, stepsize, accept_trace = accum
    beta_0, beta_1, period_elapsed = args

    log_int_1 = log_f_i(current_z, x, beta_0)
    log_int_2 = log_f_i(current_z, x, beta_1)
    log_importance_weight = log_int_2 - log_int_1

    def U(z):
      return -log_f_i(z, x, beta_1)

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

  stepsize = jnp.ones(1)*0.01
  accept_trace = jnp.zeros(1)
  current_z = random.normal(z_rng, shape=(latent_size,))
  init_accum = (current_z, stepsize, accept_trace)

  xs = (annealing_schedule[:-1], annealing_schedule[1:], jnp.arange(1,len(annealing_schedule)))

  final_accum, log_weights = jax.lax.scan(ais_step, init_accum, xs)

  return jnp.sum(log_weights)

@partial(jit, static_argnums=(1,))
def batch_ais_fn(rng, model, decoder_params, images):
  rngs = random.split(rng, len(images))
  return jnp.mean(jax.vmap(ais_trajectory, in_axes=(0, None, None, 0))(rngs, model, decoder_params, images))

@partial(jit, static_argnums=(1,2))
def ais_iwelbo_fn(rng, model,num_samples, decoder_params, images):
  rngs = random.split(rng, num_samples)
  logw_log_summand = jax.vmap(batch_ais_fn, in_axes=(0, None, None, None))( rngs, model, decoder_params, images )
  K = num_samples
  logw_iwae_K = logsumexp(logw_log_summand) - jnp.log(K)
  return logw_iwae_K
