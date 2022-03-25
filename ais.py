from jax import numpy as jnp
from jax import random, grad

from utils import log_bernoulli, log_normal
from hmc import hmc_sample_and_tune

def ais_trajectory(
  rng,
  model,
  loader,
  mode='forward',
  schedule=jnp.linspace(0, 1, 500),
  n_sample=100
):
  """Annealed importance sampling trajectories for a single batch."""
  v_rng, hmc_rng = random.split(rng)

  def intermediate_dist(z, x, beta, log_likelihood_fn=log_bernoulli):
      zeros = jnp.zeros(z.shape)
      log_prior = log_normal(z, zeros, zeros)
      log_likelihood = log_likelihood_fn(model.decode(z), x)
      return log_prior + (beta*log_likelihood)

# for i, (batch, posterior_z) in enumerate(loader):
  # ...

  batch: any
  stepsize: any
  accept_trace: any
  log_importance_weight: any

  # No need backward implementation!
  current_z: any

  log_f_i = intermediate_dist
# for period_elapsed, (t0, t1) in ....
  # ...
  period_elapsed: int
  t0: jnp.array
  t1: jnp.array

  # Log importance weight update
  log_int_1 = log_f_i(current_z, batch, t0)
  log_int_2 = log_f_i(current_z, batch, t1)
  log_importance_weight = log_importance_weight + (log_int_2 - log_int_1)

  def U(z):
      return -log_f_i(z, batch, t1)

  def grad_U(z):
      return grad(U)(z)  # WARNING :- Can be problematic? No clamp.

  def normalized_K(v):
      zeros = jnp.zeros(v.shape)
      return -log_normal(v, zeros, zeros)

  current_v = random.normal(v_rng, shape=current_z.shape)
  tuning_params = (stepsize, accept_trace, period_elapsed)

  current_z, stepsize, accept_trace = hmc_sample_and_tune(
      hmc_rng, current_z, current_v,
      U, normalized_K, grad_U,
      tuning_params
  )